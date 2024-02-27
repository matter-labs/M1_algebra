use crate::fft::bitreverse::bitreverse_enumeration_inplace;
use crate::fft::tooling::{precompute_twiddles_for_fft, distribute_powers, domain_generator_for_size, materialize_powers_parallel};
use crate::field::TwoAdicField;
use crate::m1::Mersenne31Field;
use crate::m1complex::Mersenne31Complex;
use crate::{field::Field, worker::Worker};
use crate::fft::tooling::log2_n;

/// A little bit about MersenneComplex FFT. Looks like the most efficient fields for arithmetic is Mersenne Field. 
/// In our case it is p = 2^31 - 1. However, this field if not FFt friendly: p-1 = 2^31 - 2 = 2 (2^30 - 1) 
/// – only twoadic roots. So solution is complex extention: M/(i^2+1) = {a + i*b: a,b є M}. Afyer we will get 2^31 twoadic
/// roots. Due to this we get a lot of benefits: now we can use Cooler-Tukey algorithm for complex numbers 
/// higher-radix FFTs reduced number of multiplications like radix-8 or split-radix and Two for the price of one 
/// https://www.robinscheibler.org/2013/02/13/real-fft.html
/// This FFT implemented by this article https://eprint.iacr.org/2023/824.pdf.

fn fft_with_tricks_for_mersenne(
input: &mut [Mersenne31Field],
worker: &Worker
) -> Vec<Mersenne31Complex>{
    debug_assert!(input.len().is_power_of_two());
    let mut packed = pack_real_as_complex(input);
    let forward_twiddles = precompute_twiddles_for_fft::<
    Mersenne31Complex,
    false,
    >(packed.len(), &worker);
    let log_n = log2_n(packed.len());
    serial_ct_ntt_natural_to_bitreversed(&mut packed, log_n as u32, &forward_twiddles);
    bitreverse_enumeration_inplace(&mut packed);
    let dft_res: Vec<Mersenne31Complex> = unpack_after_fft(&packed, InvFlag::Direct);
    dft_res
}

pub fn ifft_with_tricks_for_mersenne(
    input: &mut [Mersenne31Complex], 
    worker: &Worker
) -> Vec<Mersenne31Field>{

    let mut packed = idft_pack(input, InvFlag::Direct);
    // let mut packed = input;

    let log_n = packed.len().trailing_zeros();
    let twiddles = precompute_twiddles_for_fft::<
    Mersenne31Complex,
    true,
    >(packed.len(), &worker);

    serial_ct_ntt_natural_to_bitreversed(&mut packed, log_n, &twiddles);

    bitreverse_enumeration_inplace(&mut packed);

    if packed.len() > 1 {
        let n_inv = Mersenne31Complex::from_u32_with_reduction(packed.len() as u64)
            .inverse()
            .unwrap();
        let mut i = 0;
        let work_size = packed.len();
        while i < work_size {
            packed[i].mul_assign(&n_inv);
            i += 1;
        }
    }
    let res = unpack_complex_as_real(&packed);
    res


}

/// Here is the magic trick. We wanna use this algorithm https://www.robinscheibler.org/2013/02/13/real-fft.html
/// but problem is that we have one sequence. On those algorithm needed two real. 
/// So we can just create two new polynomials with using Cooler-Tukey one will be with
/// even coefficients second – odd coefficients. Now we can pack them in beautiful way.
/// And make the FFT of size n/2


// -- v[0], v[1], v[2], ... --> v[0] + I v[1], v[2] + I v[3], ...
fn pack_real_as_complex(input: &[Mersenne31Field]) -> Vec<Mersenne31Complex>{
    assert!(input.len() % 2 == 0);

    let mut even_tuple = Vec::new();
    let mut odd_tuple = Vec::new();

    for (index, &element) in input.iter().enumerate() {
        if index % 2 == 0 {
            even_tuple.push(element);
        } else {
            odd_tuple.push(element);
        }
    }

    let result: Vec<Mersenne31Complex> = even_tuple
        .iter()
        .zip(odd_tuple)
        .map(|(&x, y)| Mersenne31Complex::new(x, y)) 
        .collect();
    result
}

/// This function is applied to fft image of a packed complex vector
/// to recover fft image of the original real vector.
/// Only sends (n/2 + 1) elements as others are recoverable due to symmetry condition a_k = conj(a_{n-k})
/// We unpack the fft result with there two formulas: 
/// X[k] = (Z[k] + Z_conj[N-k])/2
/// Y[k] = -j(Z[k] - Z_conj[N-k])/2
/// Returns v[k] = X[k] + omega^k Y[k]
fn unpack_after_fft(input: &[Mersenne31Complex], inv_flag: InvFlag) -> Vec<Mersenne31Complex>{
    // let mut omega = domain_generator_for_size::<Mersenn31Complex>(input.len() as u64);
    let mut omega = Mersenne31Complex::two_adic_generator((input.len().trailing_zeros() + 1) as usize);
    match inv_flag {
        InvFlag::Direct => (),
        InvFlag::Inverse => omega = omega.inverse().unwrap(),
    };
    let mut res = Vec::with_capacity(input.len());
    let mut first_value = input[0].real_part;
    // -- This is not a bijection; but in construction we will assume this to 0 anyways, right?
    res.push( Mersenne31Complex::new(*first_value.add_assign(&input[0].imag_part), Mersenne31Field::ZERO));
    let mut omega_power = omega;
    // the vec will be in period, other word is (XN = X0)
    let k = input.len();
    for j in 1..k {
        let mut x = input[j];
        let mut y = input[k - j];

        let tmp = *y.conjugate();
        let even = x.add_assign(&tmp); // -- input[j] + conj(input[k-j])

        let mut x = input[j];
        let mut y = input[k - j];
        
        // 2 * Xo = -j * ((xr + j * xc) - (yr - j * yc)
        //        = -j * (xr - yr + j * xc + j * yc)
        let mut odd = Mersenne31Complex::new(*(x.imag_part.add_assign(&y.imag_part)), *y.real_part.sub_assign(&x.real_part));
        
        // -- (x.im + y.im) + I (y.re - x.re) =?= - I (x - conj(y)) -- seems to work
        let part = (even.add_assign(odd.mul_assign(&omega_power))).div_2exp_u64(1);
        res.push(part);
        omega_power.mul_assign(&omega);
    }
    let mut first_value = input[0].real_part;
    res.push( Mersenne31Complex::new(*first_value.sub_assign(&input[0].imag_part), Mersenne31Field::ZERO));
    res
}

// step 1 - parse our sequence as v[0] + I v[1], v[2] + I v[3], ... <-- P_complex

// a + bx + cx^2 + dx^3
//(a + cx^2) + (I/x) (bx + dx^3)
// (a+Ib) + (c+Id)x^2
// On the level of a polynomial: (P(x)+P(-x))/2 + (I/x)*(P(x)-P(-x))/ 2  =  P_complex(x^2)

// step 2 - do direct Fourier transform of P_complex -- this will output values of P_complex in roots of unity

// step 3 - recover values of P(x) :

// P_complex(omega^k), P_complex(omega^{n-k})

// X[k] = (P_complex(omega^k) + conj[P_complex](omega^k)) / 2 = (P(x)+P(-x)) / 2 | {x^2 = omega^k}
// Y[k] = 
// (P_complex(omega^k) - conj[P_complex](omega^k)) / (2I) = (omega^[-1/2])^k (P(x)-P(-x))/2 | {x^2 = omega^k}

// X[k] + (omega^[1/2])^k Y[k] = P(omega^k)

// v: [(a+c) + I(b+d)], [(a-c) + I(b-d)]

// k = 2

// the fact that X, Y here are real is a mere coincidence - normally they won't be, it is just our
// omega-value iterates only through +/- 1
// also notice that omega in the code is omega^(1/2) in my formulas

// X[0] = (v[0] + conj v[0])/2 = a + c
// Y[0] = -I (v[0] - conj v[0])/2 = b + d

// X[1] = (v[1] + conj v[1])/2 = a - c
// Y[1] = -I (v[1] - conj v[1])/2 = b - d

// RET[0] = X[0] + I^0 Y[0] = a + c + b + d
// RET[1] = X[1] + I Y[1] = (a - c) + (b - d) I

// RET[2] = X[0] + I^2 Y[0] = (a + c) - (b + d)
// RET[3] = X[1] - I^3 Y[1] = (a - c) - (b - d) I


/// Untwist.
fn idft_pack(input: &[Mersenne31Complex], inv_flag: InvFlag) -> Vec<Mersenne31Complex>{
    let len = input.len()-1;
    let mut omega = Mersenne31Complex::two_adic_generator((len.trailing_zeros() + 1) as usize);
    match inv_flag {
        InvFlag::Direct =>
            omega = omega 
                .inverse()
                .expect("must always exist for domain generator"),
        InvFlag::Inverse => (),
    }
    let mut res = Vec::with_capacity(len);
    let mut omega_power = Mersenne31Complex::ONE;
    for j in 0..len {
        let mut x = input[j];
        let mut y = input[len - j];


        let tmp = *y.conjugate();
        let even = x.add_assign(&tmp);

        let mut x = input[j];
        let mut y = input[len - j];
        
        // j * 2 * Xo = j * ((xr + j xc) - (yr - j * yc))
        //            = j * (xr - yr + j * xc + j * yc)
        //            = j * (xr - yr) - xc - yc
        //            = -(xc + yc - j * (yr -xr))
        let mut odd = Mersenne31Complex::new(*(x.imag_part.add_assign(&y.imag_part)), *y.real_part.sub_assign(&x.real_part));
        let tmp = odd.mul_assign(&omega_power);
        let tmp2 = even.sub_assign(tmp);
        let part = (tmp2).div_2exp_u64(1);
        res.push(part);
        omega_power.mul_assign(&omega);
    }
    res
}

// -- Splits the vector into complex and real parts and packs them as follows:
// -- Re(v[0]), Im(v[0]), Re(v[1]), Im(v[1]), ...
// -- Inverse to "packing" function
fn unpack_complex_as_real(input: &[Mersenne31Complex]) -> Vec<Mersenne31Field> {

    let mut even = Vec::new();
    let mut odd = Vec::new();

    for complex in input {
        even.push(complex.real_part);
        odd.push(complex.imag_part);
    }

    let mut result = Vec::with_capacity(even.len() + odd.len());

    for (even_element, odd_element) in even.into_iter().zip(odd.into_iter()) {
        result.push(even_element);
        result.push(odd_element);
    }

    result
}

pub fn ifft_natural_to_natural<E: TwoAdicField>(input: &mut [E], coset: E, twiddles: &[E]) {
    debug_assert!(input.len().is_power_of_two());
    if input.len() != 1 {
        debug_assert!(input.len() == twiddles.len() * 2);
    }

    let log_n = input.len().trailing_zeros();

    serial_ct_ntt_natural_to_bitreversed(input, log_n, twiddles);
    bitreverse_enumeration_inplace(input);

    if coset != E::ONE {
        let coset = coset.inverse().expect("inverse of coset must exist");
        distribute_powers(input, coset);
    }

    if input.len() > 1 {
        let n_inv = E::from_u32_with_reduction(input.len() as u64)
            .inverse()
            .unwrap();
        let mut i = 0;
        let work_size = input.len();
        while i < work_size {
            input[i].mul_assign(&n_inv);
            i += 1;
        }
    }
}

fn fft_naive_to_bitreversed<E: TwoAdicField>(
    input: &mut [E],
    coset: E,
    twiddles: &[E],
) {
    if input.len() != 1 {
        debug_assert!(input.len() == twiddles.len() * 2);
    }

    if coset != E::ONE {
        distribute_powers::<E>(input, coset);
    }
    let log_n = log2_n(input.len());

    serial_ct_ntt_natural_to_bitreversed(input, log_n as u32, twiddles);
}
fn naive_dft(input: &mut [Mersenne31Complex]
) -> Vec<Mersenne31Complex>{

    let omega = domain_generator_for_size::<Mersenne31Complex>(input.len() as u64);
    let mut reference = vec![];
    for i in 0..input.len() {
        let x = Field::pow_u64(&omega, i as u64); //omega.pow_u64(i as u64);
        let mut tmp = Mersenne31Complex::ZERO;
        let mut current = Mersenne31Complex::ONE;
        for coeff in input.iter() {
            let mut c = *coeff;
            Field::mul_assign(&mut c, &current); //c.mul_assign(&current);
            Field::add_assign(&mut tmp, &c); //tmp.add_assign(&c);
            Field::mul_assign(&mut current, &x); //current.mul_assign(&x);
        }

        reference.push(tmp);
    }
    reference
}

pub(crate) fn serial_ct_ntt_natural_to_bitreversed<E: TwoAdicField>(
    a: &mut [E],
    log_n: u32,
    omegas_bit_reversed: &[E],
) {
    let n = a.len();
    if n == 1 {
        return;
    }

    debug_assert!(n == omegas_bit_reversed.len() * 2);
    debug_assert!(n == (1 << log_n) as usize);

    let mut pairs_per_group = n / 2;
    let mut num_groups = 1;
    let mut distance = n / 2;

    {
        // special case for omega = 1
        debug_assert!(num_groups == 1);
        let idx_1 = 0;
        let idx_2 = pairs_per_group;

        let mut j = idx_1;

        while j < idx_2 {
            let u = a[j];
            let v = a[j + distance];

            let mut tmp = u;
            tmp.sub_assign(&v);

            a[j + distance] = tmp;
            a[j].add_assign(&v);

            j += 1;
        }

        pairs_per_group /= 2;
        num_groups *= 2;
        distance /= 2;
    }

    while num_groups < n {
        debug_assert!(num_groups > 1);
        let mut k = 0;
        while k < num_groups {
            let idx_1 = k * pairs_per_group * 2;
            let idx_2 = idx_1 + pairs_per_group;
            let s = omegas_bit_reversed[k];

            let mut j = idx_1;
            while j < idx_2 {
                let u = a[j];
                let mut v = a[j + distance];
                v.mul_assign(&s);

                let mut tmp = u;
                tmp.sub_assign(&v);

                a[j + distance] = tmp;
                a[j].add_assign(&v);

                j += 1;
            }

            k += 1;
        }

        pairs_per_group /= 2;
        num_groups *= 2;
        distance /= 2;
    }
}


/// Starts from a real valued function input on subset H of S1 of size 2^input.
/// Makes lagrange interpolation on the set of size 2^{input+resize}.
pub fn lde_naive(input: &[Mersenne31Field], resize: usize, worker: &Worker)
    ->
    Vec<Mersenne31Complex> {
    
    let size = input.len().trailing_zeros() as usize;
    assert!(input.len() == 1 << size);

    let twiddles = precompute_twiddles_for_fft::<Mersenne31Complex, true>(1 << size, worker);

    let mut state : Vec<_> = input
        .into_iter()
        .map(|x| Mersenne31Complex{real_part: *x, imag_part: Mersenne31Field::ZERO})
        .collect();

    ifft_natural_to_natural(&mut state, Mersenne31Complex::ONE, &twiddles);

    let new_size = size + resize;

    state.append(&mut vec![Mersenne31Complex::ZERO; (1 << new_size) - (1 << size)]);

    let twiddles = precompute_twiddles_for_fft::<Mersenne31Complex, false>(1 << new_size, worker);

    fft_naive_to_bitreversed(&mut state, Mersenne31Complex::ONE, &twiddles);
    
    bitreverse_enumeration_inplace(&mut state);

    state
}

#[test]
fn test_lde_naive() {

    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    
    let logsize = 10;

    let poly_size = 1 << logsize; 

    let mut acc = Mersenne31Field::ZERO;

    let mut poly = Vec::with_capacity(poly_size);
    for i in 0..poly_size-1 {
        let tmp = rand2_from_rng(&mut rng);
        poly.push(tmp);
        acc.add_assign(&tmp);
    }

    poly.push(* acc.negate()); 

    let resize = 3;

    let extended = lde_naive(&poly, resize, &worker);

    assert!(extended.len() == 1 << (resize + logsize));
    for i in 0..poly_size {
        assert!(extended[i * (1 << resize)].real_part == poly[i]);
        assert!(extended[i * (1 << resize)].imag_part == Mersenne31Field::ZERO);
    }

}


/// Returns LDE with cleared twisted factors.
/// Assumes that sum of the input is 0.
pub fn lde_twisted_naive(input: &[Mersenne31Field], resize: usize, worker: &Worker) -> Vec<Mersenne31Field> {
    let mut res = lde_naive(input, resize, worker);
    distribute_powers(&mut res, domain_generator_for_size::<Mersenne31Complex>(1 << (resize+1) as u64).inverse().unwrap());
    res.into_iter()
        .map(|Mersenne31Complex{real_part, imag_part}|{ 
            debug_assert!(imag_part == Mersenne31Field::ZERO);
            real_part
        })
        .collect()
}

#[test]
fn test_lde_twisted_naive() {

    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    
    let logsize = 9;

    let poly_size = 1 << logsize; 

    let mut acc = Mersenne31Field::ZERO;

    let mut poly = Vec::with_capacity(poly_size);
    for i in 0..poly_size-1 {
        let tmp = rand2_from_rng(&mut rng);
        poly.push(tmp);
        acc.add_assign(&tmp);
    }

    poly.push(* acc.negate()); 

    let resize = 5;

    let extended = lde_twisted_naive(&poly, resize, &worker);

}



/// Notice: this is currently not on parity with lde_twisted_naive, with odd values encoded with different sign.
/// This is not a problem, this can be chosen as canonical implementation.
/// Extracts and returns c0 separately.
pub fn lde_packed(input: &[Mersenne31Field], resize: usize, worker: &Worker) -> (Vec<Mersenne31Field>, Mersenne31Field){
    let size = input.len().trailing_zeros() as usize;
    assert!(input.len() == 1 << size);

    let coset = Mersenne31Complex::ONE;

    let mut packed_state = pack_real_as_complex(input);
    let packed_inverse_twiddles = precompute_twiddles_for_fft::<Mersenne31Complex, true>(1 << (size-1), worker);

    let packed_forward_twiddles = precompute_twiddles_for_fft::<Mersenne31Complex, false>(1 << (size+resize-1), worker);


    ifft_natural_to_natural(&mut packed_state, coset, &packed_inverse_twiddles);

    let mut state = unpack_after_fft(&packed_state, InvFlag::Inverse);
    state.iter_mut().map(|x|x.div_2exp_u64(1)).count();

// { a0 a1 a2 ... a_{n/2} (...) }

// ASSUME a0 == 0
    
    let c0_complex = state[0];
    let Mersenne31Complex{real_part: c0, imag_part} = c0_complex;
    debug_assert!(imag_part == Mersenne31Field::ZERO);
    state[0] = Mersenne31Complex::ZERO; // Nullify c0.

    assert!(state.len() == (1 << (size - 1)) + 1);
// a0 a1 ... a_j ... a_{n/2} ...conj(a_{n-j}) ... ] 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// This is not Hermitian anymore (and so its FFT is not real, but has twists).

// But we can shift this polynomial by z^{N/2 - n/2}, and it will be symmetric
// Provided a0 = 0.


// n = 1 << (size - 1)
// N = 1 << (size + resize - 1)

    let shift = (1 << (size + resize - 1)) - (1 << (size - 1)) ; // N/2 - n/2

    let mut extended_state = vec![Mersenne31Complex::ZERO; (1 << (size + resize - 1)) + 1];
    extended_state[shift .. (1 << (size + resize - 1)) + 1].copy_from_slice(&state);

    let mut packed_extended_state = idft_pack(&extended_state, InvFlag::Inverse);

    fft_naive_to_bitreversed(&mut packed_extended_state, coset, &packed_forward_twiddles);
    bitreverse_enumeration_inplace(&mut packed_extended_state);

    (unpack_complex_as_real(&packed_extended_state), c0)

}

#[test]
fn test_lde_packed() {

    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    
    let logsize = 9;

    let poly_size = 1 << logsize; 

    let mut acc = Mersenne31Field::ZERO;

    let mut poly = Vec::with_capacity(poly_size);
    for i in 0..poly_size-1 {
        let tmp = rand2_from_rng(&mut rng);
        poly.push(tmp);
        acc.add_assign(&tmp);
    }

    poly.push(* acc.negate()); 

    let resize = 5;

    let extended_naive = lde_twisted_naive(&poly, resize, &worker);
    let (extended_packed, should_be_zero) = lde_packed(&poly, resize, &worker);

    assert!(should_be_zero == Mersenne31Field::ZERO);

    assert!(extended_naive.len() == extended_packed.len());

    extended_naive.iter().zip(extended_packed.iter()).enumerate().map(|(i,(x, y))| {
        assert!(if i%2 == 0 {x == y} else {* x.clone().add_assign(y) == Mersenne31Field::ZERO})
    }).count();

}
pub fn lde_compress(
    input: &mut [Mersenne31Field],
    resize: usize,
    worker: &Worker
) -> (Vec<Mersenne31Field>, Mersenne31Complex, usize){
    let base_bits = log2_n(input.len());

    assert!((base_bits + resize) <= 31);

    let mut w = fft_with_tricks_for_mersenne(input ,worker);
    let poly_size = w.len();
    let c0 = w[0].div_2exp_u64(base_bits as u64);

    w[0] = Mersenne31Complex::ZERO;

    let mut omega = Mersenne31Complex::two_adic_generator(base_bits + resize)
        .inverse().expect("must always exist for domain generator");


    let num_powers = 1 << resize;
    let mut powers = materialize_powers_parallel(omega, num_powers, worker);
    bitreverse_enumeration_inplace(&mut powers);

    let lde_res: Vec<Vec<Mersenne31Field>> = powers.iter().map(|t: &Mersenne31Complex| {
        // t^(k-H/2) where k = 1, ..., H-1
        // to simplify evaluation: t^k * t^-H/2
        let mut t_pow_h_div_2 = t.clone();
        let t_pow_h_div_2 = t_pow_h_div_2.exp_power_of_2(base_bits - 1).inverse().unwrap();
        let mut t_pow_k = materialize_powers_parallel(*t, poly_size, worker);
        bitreverse_enumeration_inplace(&mut t_pow_k);
        let shift_factor:Vec<Mersenne31Complex> = t_pow_k.iter().map(|a| {
            let mut a = a.clone();
            a.mul_assign(&t_pow_h_div_2);
            a
        }).collect();
        let mut equation: Vec<Mersenne31Complex> = w.iter().zip(&shift_factor).map(|(elem, shift)| {
            let mut e = elem.clone();
            e.mul_assign(&shift);
            e
        }).collect();
        ifft_with_tricks_for_mersenne(&mut equation, worker)
    }).collect();

    let mut res: Vec<Mersenne31Field> = Vec::with_capacity(1 << (base_bits + resize));
    for i in 0..(1 << base_bits) {
        res.extend(lde_res.iter().map(| x| x[i]))
    } 

    (res, c0, base_bits )
}

use itertools::Itertools;
pub fn lde_decompress(input: &[Mersenne31Field], c0: Mersenne31Complex, h: usize, worker: &Worker) -> Vec<Mersenne31Complex>{
    // p(x) = phi(t)*F + c0
    let bits = log2_n(input.len());
    let resize_bits = bits - h;

    // phi(t) = t^H/2
    let omega = Mersenne31Complex::two_adic_generator(bits);
    let num_powers = 1 << resize_bits;
    let mut powers = materialize_powers_parallel(omega, num_powers, worker);
    bitreverse_enumeration_inplace(&mut powers);
    let t: Vec<Mersenne31Complex> = powers.iter().map(|x| {
        let mut t = x.clone();
        let t = t.exp_power_of_2(h - 1);
        t
    }).collect();
    let mut res = Vec::with_capacity(input.len());
    for chunk in input.iter().chunks(1 << resize_bits).into_iter(){
        res.extend(chunk.into_iter().zip(&t).map(|(elem, t)| {
            let mut tmp = t.clone();
            tmp.mul_assign(&Mersenne31Complex::new_from_real(*elem));
            tmp.add_assign(&c0);
            tmp
    
        }));
    }

    res
}

#[test]
fn lde_compress_test()
{
    const ADDED_BITS: usize = 1;
    let worker = Worker::new();
    let value = [Mersenne31Field::ONE, Mersenne31Field::TWO, Mersenne31Field::ZERO, Mersenne31Field::MINUS_ONE].to_vec();
    let mut input_real = value.clone();
    let expected_values = 
        [
            Mersenne31Complex::new_from_real(Mersenne31Field::ONE),
            Mersenne31Complex::new(Mersenne31Field::new(1073741824), Mersenne31Field::new(32768)),
            Mersenne31Complex::new_from_real(Mersenne31Field::TWO),
            Mersenne31Complex::new(Mersenne31Field::new(1073741824), Mersenne31Field::new(2147418111)),
            Mersenne31Complex::new_from_real(Mersenne31Field::ZERO),
            Mersenne31Complex::new(Mersenne31Field::new(1073741824), Mersenne31Field::new(2147450879)),
            Mersenne31Complex::new_from_real(Mersenne31Field::MINUS_ONE),
            Mersenne31Complex::new(Mersenne31Field::new(1073741824), Mersenne31Field::new(65536)),
        ]
        .to_vec();

    // let values = [
    //     Mersenne31Field::new(1192083057),
    //     Mersenne31Field::new(1317644184),
    //     Mersenne31Field::new(1777280870),
    //     Mersenne31Field::new(5599001), 
    //     Mersenne31Field::new(614669447), 
    //     Mersenne31Field::new(1350110858),
    //     Mersenne31Field::new(293550253),
    //     Mersenne31Field::new(232212973),
    // ].to_vec();
    let (compressed, c0, h) = lde_compress(&mut input_real, ADDED_BITS, &worker);
    dbg!(compressed.clone());
    let output = lde_decompress(&compressed, c0, h, &worker);
    dbg!(output.clone());

    assert_eq!(expected_values, output);


}


// #[test]

// fn test_lde_compress() {
//     let worker = Worker::new();
//     let mut rng = rand::thread_rng();
    
//     let logsize = 9;

//     let poly_size = 1 << logsize; 

//     let mut acc = Mersenne31Field::ZERO;

//     let mut poly = Vec::with_capacity(poly_size);
//     for i in 0..poly_size-1 {
//         let tmp = rand2_from_rng(&mut rng);
//         poly.push(tmp);
//         acc.add_assign(&tmp);
//     }

//     poly.push(* acc.negate()); 

//     let resize = 5;

//     let extended1 = lde_twisted_naive(&poly, resize, &worker);

//     let (extended2, c0, _) = lde_compress(&mut poly.clone(), resize, &worker);

//     assert!(c0 == Mersenne31Complex::ZERO);
//     assert!(extended1.len() == extended2.len());
//     extended1.iter().zip(extended2.iter()).map(|(lhs, rhs)|assert!(*lhs == *rhs)).count();
// }


pub fn rand_from_rng<R: rand::Rng>(rng: &mut R) -> Mersenne31Complex {
    let a = Mersenne31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    let b = Mersenne31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    Mersenne31Complex::new(a, b)

}
pub fn rand2_from_rng<R: rand::Rng>(rng: &mut R) -> Mersenne31Field {
    let a = Mersenne31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    a
}

pub enum InvFlag {
    Direct,
    Inverse,
}


#[test]
fn test_over_merssenecomplex_naive() {
    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 0..10 {
        let poly_size = 1 << poly_size_log;

        let original: Vec<Mersenne31Complex> =
            (0..poly_size).map(|_| rand_from_rng(&mut rng)).collect();

        let forward_twiddles = precompute_twiddles_for_fft::<
            Mersenne31Complex,
            false,
        >(original.len(), &worker);
        let mut forward = original.clone();
        fft_naive_to_bitreversed(&mut forward, Mersenne31Complex::ONE, &forward_twiddles[..]);
        bitreverse_enumeration_inplace(&mut forward);

        let omega = domain_generator_for_size::<Mersenne31Complex>(poly_size as u64);
        let mut reference = vec![];
        for i in 0..poly_size {
            let x = Field::pow_u64(&omega, i as u64); //omega.pow_u64(i as u64);
            let mut tmp = Mersenne31Complex::ZERO;
            let mut current = Mersenne31Complex::ONE;
            for coeff in original.iter() {
                let mut c = *coeff;
                Field::mul_assign(&mut c, &current); //c.mul_assign(&current);
                Field::add_assign(&mut tmp, &c); //tmp.add_assign(&c);
                Field::mul_assign(&mut current, &x); //current.mul_assign(&x);
            }

            reference.push(tmp);
        }

        assert_eq!(reference, forward, "failed for size 2^{}", poly_size_log);
    }
}
#[test]
fn test_over_mersenncomplex_inverse() {
    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 1..20 {
        let poly_size = 1 << poly_size_log;

        let original: Vec<Mersenne31Complex> =
            (0..poly_size).map(|_| rand_from_rng(&mut rng)).collect();
        let inverse_twiddles = precompute_twiddles_for_fft::<
            Mersenne31Complex,
            true,
        >(poly_size, &worker);
        let forward_twiddles = precompute_twiddles_for_fft::<
            Mersenne31Complex,
            false,
        >(poly_size, &worker);

        let mut forward = original.clone();
        fft_naive_to_bitreversed(&mut forward, Mersenne31Complex::ONE, &forward_twiddles[..]);
        bitreverse_enumeration_inplace(&mut forward);

        let mut back = forward.clone();
        ifft_natural_to_natural(&mut back, Mersenne31Complex::ONE, &inverse_twiddles[..]);

        assert_eq!(original, back, "failed for size 2^{}", poly_size_log);
    }
}
#[test]
fn test_over_mersennetrick_inverse() {
    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 1..10 {
        let poly_size = 1 << poly_size_log; 

        let original: Vec<Mersenne31Field> =
            (0..poly_size).map(|_| rand2_from_rng(&mut rng)).collect();

        let mut forward = original.clone();
        let mut fft_res = fft_with_tricks_for_mersenne(&mut forward, &worker);
        // bitreverse_enumeration_inplace(&mut fft_res);
        let ifft_res = ifft_with_tricks_for_mersenne(&mut fft_res, &worker);
        assert_eq!(original.to_vec(), ifft_res, "failed for size 2^{}", poly_size_log);
    }
}
#[test]
fn test_over_mersennetrick(){

    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 1..10 {
        let poly_size = 1 << poly_size_log; 
        let mut a: Vec<Mersenne31Field> =
            (0..poly_size).map(|_| rand2_from_rng(&mut rng)).collect();
        
        let mut b: Vec<Mersenne31Field> =
            (0..poly_size).map(|_| rand2_from_rng(&mut rng)).collect();
        let mut a_copy = a.clone();
        let b_copy = b.clone();
    
        let mut fft_a = fft_with_tricks_for_mersenne(&mut a, &worker);
        let fft_b = fft_with_tricks_for_mersenne(&mut b, &worker);

        let mut fft_c: Vec<Mersenne31Complex> = vec![];
        for (i, j) in fft_a.iter().zip(fft_b.iter()){
            let mut a = *i;
            a.mul_assign(&j);
            fft_c.push(a)
        }

        let c = ifft_with_tricks_for_mersenne(&mut fft_c, &worker);
        let mut res = Vec::with_capacity(poly_size);
        for i in 0..poly_size {
            let mut tmp = Mersenne31Field::ZERO;
            for j in 0..poly_size {
                let mut x = a_copy[j];
                x.mul_assign(&b_copy[(poly_size + i - j) % poly_size]);
                tmp.add_assign(&x);
            }
            res.push(tmp);
        }
    
        assert_eq!(c, res);
    }
}

