use crate::fft::bitreverse::bitreverse_enumeration_inplace;
use crate::fft::tooling::log2_n;
use crate::fft::tooling::{
    distribute_powers, domain_generator_for_size, materialize_powers_parallel,
    precompute_twiddles_for_fft,
};
use crate::field::TwoAdicField;
use crate::m1::Mersenne31Field;
use crate::m1complex::Mersenne31Complex;
use crate::{field::Field, worker::Worker};

type M31F = Mersenne31Field;
type M31C = Mersenne31Complex;

// Roots of unity needed by size_8_dit
// Friendlier values could be obtained with a different 2-adic generator.
const W_1_8: M31C = M31C::new(M31F::new(32768), M31F::new(2147450879));
const W_1_4: M31C = M31C::new(M31F::new(0), M31F::new(2147483646));
const W_3_8: M31C = M31C::new(M31F::new(2147450879), M31F::new(2147450879));

// Optimized special-case math could go here.
fn mul_assign_w_1_8(x: &mut M31C) {
    x.mul_assign(&W_1_8);
}

fn mul_assign_w_1_4(x: &mut M31C) {
    x.mul_assign(&W_1_4);
}

fn mul_assign_w_3_8(x: &mut M31C) {
    x.mul_assign(&W_3_8);
}

// Like boojum-cuda, uses DIT for natural->bitrev.
// Obviously the math here isn't optimal,
// i'm just trying to get the right dataflow first.
fn size_8_radix_2_dit(x: &mut [M31C; 8]) {
    // first stage
    for i in 0..4 {
        let mut tmp = x[i as usize];
        x[i] = *tmp.clone().add_assign(&x[i + 4]);
        x[i + 4] = *tmp.sub_assign(&x[i + 4]);
    }

    // second stage
    mul_assign_w_1_4(&mut x[6]);
    mul_assign_w_1_4(&mut x[7]);
    for r in [0..2, 4..6].iter() {
        for i in r.clone().into_iter() {
            let mut tmp = x[i];
            x[i] = *tmp.clone().add_assign(&x[i + 2]);
            x[i + 2] = *tmp.sub_assign(&x[i + 2]);
        }
    }

    // third stage
    mul_assign_w_1_4(&mut x[3]);
    mul_assign_w_1_8(&mut x[5]);
    mul_assign_w_3_8(&mut x[7]);
    for i in [0, 2, 4, 6].iter() {
        let i = *i as usize;
        let mut tmp = x[i];
        x[i] = *tmp.clone().add_assign(&x[i + 1]);
        x[i + 1] = *tmp.sub_assign(&x[i + 1]);
    }

    // undo bitrev
    let tmp = x[1];
    x[1] = x[4];
    x[4] = tmp;
    let tmp = x[3];
    x[3] = x[6];
    x[6] = tmp;
}

fn stockham_radix_8_dif_impl(
    x: &mut [M31C],
    y: &mut [M31C],
    independent_fft_count: usize,
    independent_fft_len: usize,
    twiddles: &[M31C],
) {
    // radix-8 exchanges (same pattern regardless of stage)
    let len_over_8 = x.len() / 8;
    let mut scratch = [M31C::ZERO; 8];
    for i in 0..len_over_8 {
        // can't collect into a fixed-size array
        // let mut size_8_fft: Vec<M31C> = x[i..]
        //     .iter()
        //     .step_by(P)
        //     .take(8)
        //     .map(|x| x.clone())
        //     .collect();
        for j in 0..8 {
            scratch[j] = x[i + len_over_8 * j];
        }
        size_8_radix_2_dit(&mut scratch);
        for j in 0..8 {
            x[i + len_over_8 * j] = scratch[j];
        }
    }

    if independent_fft_len == 8 {
        return;
    }

    // apply twiddles
    let len_over_8 = independent_fft_len / 8;
    let shift = x.len() / independent_fft_len;
    for i in 1..8 {
        for j in 1..len_over_8 {
            let twiddle = twiddles[i * j * shift];
            for k in 0..independent_fft_count {
                x[k + independent_fft_count * (j + len_over_8 * i)].mul_assign(&twiddle);
            }
        }
    }

    // transpose within independent ffts
    for i in 0..8 {
        for j in 0..len_over_8 {
            for k in 0..independent_fft_count {
                y[k + independent_fft_count * (i + 8 * j)] =
                    x[k + independent_fft_count * (j + len_over_8 * i)];
            }
        }
    }

    stockham_radix_8_dif_impl(
        y,
        x,
        independent_fft_count * 8,
        independent_fft_len / 8,
        twiddles,
    );
}

pub fn stockham_radix_8_dif(x: &mut [M31C], y: &mut [M31C], twiddles: &[M31C]) {
    stockham_radix_8_dif_impl(x, y, 1, x.len(), twiddles);
}

#[test]
fn test_compare() {
    use crate::fft::fft::{rand_from_rng, serial_ct_ntt_natural_to_bitreversed};

    let worker = Worker::new();
    let mut rng = rand::thread_rng();

    let log_n = 6;
    let n = 1 << log_n;

    let mut x: Vec<M31C> = (0..n).map(|_| rand_from_rng(&mut rng)).collect();

    let twiddles = precompute_twiddles_for_fft::<
        M31C,
        false,
    >(x.len(), &worker);
    let mut reference = x.clone();
    serial_ct_ntt_natural_to_bitreversed(
        &mut reference,
        log_n,
        &twiddles[..]
    );
    bitreverse_enumeration_inplace(&mut reference);

    let mut twiddles = vec![M31C::ONE; n];
    let omega = domain_generator_for_size::<M31C>(n as u64);
    distribute_powers(&mut twiddles, omega);
    let mut y = vec![M31C::ZERO; n];
    stockham_radix_8_dif(&mut x, &mut y, &twiddles);

    // for (i, (output, control)) in x.iter().zip(&reference).enumerate() {
    //     println!("{} {}", *output, control);
    // }
    for (i, (output, control)) in y.iter().zip(reference).enumerate() {
        assert_eq!(*output, control, "failed at {}", i);
    }
}
