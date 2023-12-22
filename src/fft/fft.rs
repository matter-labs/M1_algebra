use crate::fft::bitreverse::bitreverse_enumeration_inplace;
use crate::fft::tooling::{precompute_twiddles_for_fft, distribute_powers, domain_generator_for_size};
use crate::field::TwoAdicField;
use crate::m1::Mersenn31Field;
use crate::m1complex::Mersenn31Complex;
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
input: &mut [Mersenn31Field],
worker: &Worker
) -> Vec<Mersenn31Complex>{
    debug_assert!(input.len().is_power_of_two());
    let mut packed = packing(input);
    let forward_twiddles = precompute_twiddles_for_fft::<
    Mersenn31Complex,
    false,
    >(packed.len(), &worker);
    let log_n = log2_n(packed.len());
    serial_ct_ntt_natural_to_bitreversed(&mut packed, log_n as u32, &forward_twiddles);
    bitreverse_enumeration_inplace(&mut packed);
    let dft_res: Vec<Mersenn31Complex> = unpack(&packed);
    dft_res
}
pub fn ifft_with_tricks_for_mersenne(
    input: &mut [Mersenn31Complex], 
    worker: &Worker
) -> Vec<Mersenn31Field>{

    let mut packed = idft_pack(input);

    let log_n = packed.len().trailing_zeros();
    let twiddles = precompute_twiddles_for_fft::<
    Mersenn31Complex,
    true,
    >(packed.len(), &worker);

    serial_ct_ntt_natural_to_bitreversed(&mut packed, log_n, &twiddles);

    bitreverse_enumeration_inplace(&mut packed);

    if packed.len() > 1 {
        let n_inv = Mersenn31Complex::from_u32_with_reduction(packed.len() as u64)
            .inverse()
            .unwrap();
        let mut i = 0;
        let work_size = packed.len();
        while i < work_size {
            packed[i].mul_assign(&n_inv);
            i += 1;
        }
    }
    let res = idft_unpack(&packed);
    res


}


/// Here is the magic trick. We wanna use this algorithm https://www.robinscheibler.org/2013/02/13/real-fft.html
/// but problem is that we have one sequence. On those algorithm needed two real. 
/// So we can just create two new polynomials with using Cooler-Tukey one will be with
/// even coefficients second – odd coefficients. Now we can packing them in beautiful way.
/// And make the FFT of size n/2

fn packing(input: &[Mersenn31Field]) -> Vec<Mersenn31Complex>{
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

    let result: Vec<Mersenn31Complex> = even_tuple
        .iter()
        .zip(odd_tuple)
        .map(|(&x, y)| Mersenn31Complex::new(x, y)) 
        .collect();
    result
}
/// We unpack the fft result with there two formulas: 
/// X[k] = (Z[k] + Z_conj[N-k])/2
/// Y[k] = -j(Z[k] - Z_conj[N-k])/2
/// Base sequances 
fn unpack(input: &[Mersenn31Complex]) -> Vec<Mersenn31Complex>{
    // let mut omega = domain_generator_for_size::<Mersenn31Complex>(input.len() as u64);
    let mut omega = Mersenn31Complex::two_adic_generator((input.len().trailing_zeros() + 1) as usize);
    let mut res = Vec::with_capacity(input.len());
    let mut first_value = input[0].real_part;
    res.push( Mersenn31Complex::new(*first_value.add_assign(&input[0].imag_part), Mersenn31Field::ZERO));
    let mut omega_power = omega;
    // the vec will be in period, other word is (XN = X0)
    let k = input.len();
    for j in 1..k {
        let mut x = input[j];
        let mut y = input[k - j];

        let tmp = *y.conjugate();
        let even = x.add_assign(&tmp);

        let mut x = input[j];
        let mut y = input[k - j];
        
        let mut odd = Mersenn31Complex::new(*(x.imag_part.add_assign(&y.imag_part)), *y.real_part.sub_assign(&x.real_part));
        let part = (even.add_assign(odd.mul_assign(&omega_power))).div_2exp_u64(1);
        res.push(part);
        omega_power.mul_assign(&omega);
    }
    let mut first_value = input[0].real_part;
    res.push( Mersenn31Complex::new(*first_value.sub_assign(&input[0].imag_part), Mersenn31Field::ZERO));
    res
}

fn idft_pack(input: &[Mersenn31Complex]) -> Vec<Mersenn31Complex>{
    let len = input.len()-1;
    let mut omega = Mersenn31Complex::two_adic_generator((len.trailing_zeros() + 1) as usize);
    omega = omega 
        .inverse()
        .expect("must always exist for domain generator");

    let mut res = Vec::with_capacity(len);
    let mut omega_power = Mersenn31Complex::ONE;
    for j in 0..len {
        let mut x = input[j];
        let mut y = input[len - j];


        let tmp = *y.conjugate();
        let even = x.add_assign(&tmp);

        let mut x = input[j];
        let mut y = input[len - j];
        
        let mut odd = Mersenn31Complex::new(*(x.imag_part.add_assign(&y.imag_part)), *y.real_part.sub_assign(&x.real_part));
        let tmp = odd.mul_assign(&omega_power);
        let tmp2 = even.sub_assign(tmp);
        let part = (tmp2).div_2exp_u64(1);
        res.push(part);
        omega_power.mul_assign(&omega);
    }
    res
}
fn idft_unpack(input: &[Mersenn31Complex]) -> Vec<Mersenn31Field> {

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

fn fft_lde(input: &mut [Mersenn31Complex], resize_bit: usize, coset: Mersenn31Complex, worker: &Worker) -> Vec<Mersenn31Complex> {
    let mut coeff: Vec<Mersenn31Field> = ifft_with_tricks_for_mersenne(input, &worker);
    coeff.resize(coeff.len() << resize_bit, Mersenn31Field::ZERO);
    let mut new_input_vec: Vec<Mersenn31Complex> = coeff.into_iter().map(Mersenn31Complex::new_from_real).collect();
    let new_input_slice = new_input_vec.as_mut_slice();
    if coset != Mersenn31Complex::ONE {
        distribute_powers::<Mersenn31Complex>(new_input_slice, coset);
    }
    let forward_twiddles = precompute_twiddles_for_fft::<
    Mersenn31Complex,
    false,
    >(new_input_slice.len(), &worker);
    let log_n = log2_n(new_input_slice.len());
    serial_ct_ntt_natural_to_bitreversed(new_input_slice, log_n as u32, &forward_twiddles);
    bitreverse_enumeration_inplace(new_input_slice);
    let dft_res: Vec<Mersenn31Complex> = unpack(&new_input_slice);
    dft_res
}

pub fn rand_from_rng<R: rand::Rng>(rng: &mut R) -> Mersenn31Complex {
    let a = Mersenn31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    let b = Mersenn31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    Mersenn31Complex::new(a, b)

}
pub fn rand2_from_rng<R: rand::Rng>(rng: &mut R) -> Mersenn31Field {
    let a = Mersenn31Field::from_u64_unchecked(rng.gen_range(0..((1 << 31) - 1)));
    a
}


// #[test]
// fn test_coset() {
//     let worker = Worker::new();
//     let mut rng = rand::thread_rng();
//     for poly_size_log in 1..10 {
//         let poly_size = 1 << poly_size_log;

//         let original: Vec<Mersenn31Complex> =
//             (0..poly_size).map(|_| rand_from_rng(&mut rng)).collect();

//         let forward_twiddles = precompute_twiddles_for_fft::<
//             Mersenn31Complex,
//             false,
//         >(original.len(), &worker);
//         let mut forward = original.clone();
//         let coset = Mersenn31Complex::generator();
//         let res = fft_lde(&mut forward, 1, coset, &worker);

//         dbg!(res);
//     }
// }

#[test]
fn test_over_merssenecomplex_naive() {
    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 0..10 {
        let poly_size = 1 << poly_size_log;

        let original: Vec<Mersenn31Complex> =
            (0..poly_size).map(|_| rand_from_rng(&mut rng)).collect();

        let forward_twiddles = precompute_twiddles_for_fft::<
            Mersenn31Complex,
            false,
        >(original.len(), &worker);
        let mut forward = original.clone();
        fft_naive_to_bitreversed(&mut forward, Mersenn31Complex::ONE, &forward_twiddles[..]);
        bitreverse_enumeration_inplace(&mut forward);

        let omega = domain_generator_for_size::<Mersenn31Complex>(poly_size as u64);
        let mut reference = vec![];
        for i in 0..poly_size {
            let x = Field::pow_u64(&omega, i as u64); //omega.pow_u64(i as u64);
            let mut tmp = Mersenn31Complex::ZERO;
            let mut current = Mersenn31Complex::ONE;
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

        let original: Vec<Mersenn31Complex> =
            (0..poly_size).map(|_| rand_from_rng(&mut rng)).collect();
        let inverse_twiddles = precompute_twiddles_for_fft::<
            Mersenn31Complex,
            true,
        >(poly_size, &worker);
        let forward_twiddles = precompute_twiddles_for_fft::<
            Mersenn31Complex,
            false,
        >(poly_size, &worker);

        let mut forward = original.clone();
        fft_naive_to_bitreversed(&mut forward, Mersenn31Complex::ONE, &forward_twiddles[..]);
        bitreverse_enumeration_inplace(&mut forward);

        let mut back = forward.clone();
        ifft_natural_to_natural(&mut back, Mersenn31Complex::ONE, &inverse_twiddles[..]);

        assert_eq!(original, back, "failed for size 2^{}", poly_size_log);
    }
}
#[test]
fn test_over_mersennetrick_inverse() {
    let worker = Worker::new();
    let mut rng = rand::thread_rng();
    for poly_size_log in 1..10 {
        let poly_size = 1 << poly_size_log; 

        let original: Vec<Mersenn31Field> =
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
        let mut a: Vec<Mersenn31Field> =
            (0..poly_size).map(|_| rand2_from_rng(&mut rng)).collect();
        
        let mut b: Vec<Mersenn31Field> =
            (0..poly_size).map(|_| rand2_from_rng(&mut rng)).collect();
        let mut a_copy = a.clone();
        let b_copy = b.clone();
    
        let mut fft_a = fft_with_tricks_for_mersenne(&mut a, &worker);
        let fft_b = fft_with_tricks_for_mersenne(&mut b, &worker);

        let mut fft_c: Vec<Mersenn31Complex> = vec![];
        for (i, j) in fft_a.iter().zip(fft_b.iter()){
            let mut a = *i;
            a.mul_assign(&j);
            fft_c.push(a)
        }

        let c = ifft_with_tricks_for_mersenne(&mut fft_c, &worker);
        let mut res = Vec::with_capacity(poly_size);
        for i in 0..poly_size {
            let mut tmp = Mersenn31Field::ZERO;
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
