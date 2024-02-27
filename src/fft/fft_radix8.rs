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
fn size_2_radix_2_dit(x: &mut [M31C; 8]) {
    let mut tmp = x[0];
    x[0] = *tmp.clone().add_assign(&x[1]);
    x[1] = *tmp.sub_assign(&x[1]);
}

fn size_4_radix_2_dit(x: &mut [M31C; 8]) {
    // first stage
    for i in 0..2 {
        let mut tmp = x[i as usize];
        x[i] = *tmp.clone().add_assign(&x[i + 2]);
        x[i + 2] = *tmp.sub_assign(&x[i + 2]);
    }

    // second stage
    mul_assign_w_1_4(&mut x[3]);
    for i in [0, 2].iter() {
        let i = *i as usize;
        let mut tmp = x[i];
        x[i] = *tmp.clone().add_assign(&x[i + 1]);
        x[i + 1] = *tmp.sub_assign(&x[i + 1]);
    }

    // undo bitrev
    let tmp = x[1];
    x[1] = x[2];
    x[2] = tmp;
}

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

fn stockham_radix_8_dif_naive_impl(
    x: &mut [M31C],
    y: &mut [M31C],
    independent_fft_count: usize,
    independent_fft_len: usize,
    twiddles: &[M31C],
    eo: bool,
) {
    // radix <= 8 exchanges
    let sub_dft_len = std::cmp::min(8, independent_fft_len);
    let sub_dft_count = x.len() / sub_dft_len;
    let mut scratch = [M31C::ZERO; 8]; // always 8, must be constant
    for i in 0..sub_dft_count {
        // can't collect into a fixed-size array
        // let mut size_8_fft: Vec<M31C> = x[i..]
        //     .iter()
        //     .step_by(P)
        //     .take(8)
        //     .map(|x| x.clone())
        //     .collect();
        for j in 0..sub_dft_len {
            scratch[j] = x[i + sub_dft_count * j];
        }
        // dit works for sub-dfts even though high-level algo is dif
        match sub_dft_len {
            8 => size_8_radix_2_dit(&mut scratch),
            4 => size_4_radix_2_dit(&mut scratch),
            2 => size_2_radix_2_dit(&mut scratch),
            _ => (),
        }
        if eo && independent_fft_len <= 8 {
            for j in 0..sub_dft_len {
                y[i + sub_dft_count * j] = scratch[j];
            }
        } else {
            for j in 0..sub_dft_len {
                x[i + sub_dft_count * j] = scratch[j];
            }
        }
    }

    if independent_fft_len <= 8 {
        return;
    }

    // apply twiddles
    let len_over_8 = independent_fft_len / 8;
    let shift = x.len() / independent_fft_len;
    for i in 1..8 {
        // let twiddle_i = twiddles[i * shift];
        // let mut twiddle_i_j = twiddle_i;
        for j in 1..len_over_8 {
            let twiddle_i_j = twiddles[i * j * shift]; // faster, believe it or not
            for k in 0..independent_fft_count {
                x[k + independent_fft_count * (j + len_over_8 * i)].mul_assign(&twiddle_i_j);
            }
            // twiddle_i_j.mul_assign(&twiddle_i);
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

    stockham_radix_8_dif_naive_impl(
        y,
        x,
        independent_fft_count * 8,
        independent_fft_len / 8,
        twiddles,
        !eo,
    );
}

pub fn stockham_radix_8_dif_naive(x: &mut [M31C], y: &mut [M31C], twiddles: &[M31C]) {
    stockham_radix_8_dif_naive_impl(x, y, 1, x.len(), twiddles, true);
}

fn stockham_radix_8_dif_cache_blocked_impl(
    x: &mut [M31C],
    y: &mut [M31C],
    independent_fft_count: usize,
    independent_fft_len: usize,
    twiddles: &[M31C],
    eo: bool,
) {
    // radix <= 8 exchanges
    let mut scratch = [M31C::ZERO; 8]; // always 8, must be constant

    if independent_fft_len <= 8 {
        let sub_dft_len = std::cmp::min(8, independent_fft_len);
        let sub_dft_count = x.len() / sub_dft_len;
        for i in 0..sub_dft_count {
            for j in 0..sub_dft_len {
                scratch[j] = x[i + sub_dft_count * j];
            }
            match sub_dft_len {
                8 => size_8_radix_2_dit(&mut scratch),
                4 => size_4_radix_2_dit(&mut scratch),
                2 => size_2_radix_2_dit(&mut scratch),
                _ => (),
            }
            if eo && independent_fft_len <= 8 {
                for j in 0..sub_dft_len {
                    y[i + sub_dft_count * j] = scratch[j];
                }
            } else {
                for j in 0..sub_dft_len {
                    x[i + sub_dft_count * j] = scratch[j];
                }
            }
        }
        return;
    }

    // do sub-dfts, apply twiddles, and transpose within independent ffts
    // attempt cache blocking in j and k
    let len_over_8 = independent_fft_len / 8;
    let shift = x.len() / independent_fft_len;
    for i in 0..len_over_8 {
        for j in 0..independent_fft_count {
            for k in 0..8 {
                scratch[k] = x[j + independent_fft_count * (i + len_over_8 * k)];
                // y[j + independent_fft_count * (i + len_over_8 * k)] = x[j + independent_fft_count * (i + len_over_8 * k)];
                // y[j + independent_fft_count * (k + 8 * i)] = x[j + independent_fft_count * (k + 8 * i)];
            }
            size_8_radix_2_dit(&mut scratch);
            for k in 0..8 {
                if i > 0 && k > 0 {
                    let twiddle_i_k = twiddles[i * k * shift];
                    scratch[k].mul_assign(&twiddle_i_k);
                }
                y[j + independent_fft_count * (k + 8 * i)] = scratch[k];
            }
        }
    }

    stockham_radix_8_dif_cache_blocked_impl(
        y,
        x,
        independent_fft_count * 8,
        independent_fft_len / 8,
        twiddles,
        !eo,
    );
}

pub fn stockham_radix_8_dif_cache_blocked(x: &mut [M31C], y: &mut [M31C], twiddles: &[M31C]) {
    stockham_radix_8_dif_cache_blocked_impl(x, y, 1, x.len(), twiddles, true);
}

fn stockham_radix_8_dit_naive_non_8_last_impl(
    x: &mut [M31C],
    y: &mut [M31C],
    independent_fft_count: usize,
    independent_fft_len: usize,
    twiddles: &[M31C],
    eo: bool,
) {
    if independent_fft_len > 8 {
        stockham_radix_8_dit_naive_non_8_last_impl(
            y,
            x,
            independent_fft_count * 8,
            independent_fft_len / 8,
            twiddles,
            !eo,
        );

        // apply twiddles
        let len_over_8 = independent_fft_len / 8;
        let shift = x.len() / independent_fft_len;
        for j in 1..len_over_8 {
            for i in 1..8 {
                let twiddle_i_j = twiddles[i * j * shift];
                for k in 0..independent_fft_count {
                    x[k + independent_fft_count * (i + 8 * j)].mul_assign(&twiddle_i_j);
                }
            }
        }
    }

    // radix <= 8 exchanges
    let sub_dft_len = std::cmp::min(8, independent_fft_len);
    let sub_dft_regions = x.len() / sub_dft_len / independent_fft_count;
    let sub_region_len = sub_dft_len * independent_fft_count;
    let mut scratch = [M31C::ZERO; 8]; // always 8, must be constant
    for i in 0..sub_dft_regions {
        for j in 0..independent_fft_count {
            if !eo && independent_fft_len <= 8 {
                for k in 0..sub_dft_len {
                    scratch[k] = y[j + independent_fft_count * k + sub_region_len * i];
                }
            } else {
                for k in 0..sub_dft_len {
                    scratch[k] = x[j + independent_fft_count * k + sub_region_len * i];
                }
            }
            match sub_dft_len {
                8 => size_8_radix_2_dit(&mut scratch),
                4 => size_4_radix_2_dit(&mut scratch),
                2 => size_2_radix_2_dit(&mut scratch),
                _ => (),
            }
            if independent_fft_len <= 8 {
                for k in 0..sub_dft_len {
                    y[j + independent_fft_count * k + sub_region_len * i] = scratch[k];
                }
            } else {
                for k in 0..sub_dft_len {
                    x[j + independent_fft_count * k + sub_region_len * i] = scratch[k];
                }
            }
        }
    }

    if independent_fft_len <= 8 {
        return;
    }

    // transpose within independent ffts
    let len_over_8 = independent_fft_len / 8;
    for j in 0..len_over_8 {
        for i in 0..8 {
            for k in 0..independent_fft_count {
                y[k + independent_fft_count * (j + len_over_8 * i)] =
                    x[k + independent_fft_count * (i + 8 * j)];
            }
        }
    }
}

pub fn stockham_radix_8_dit_naive_non_8_last(x: &mut [M31C], y: &mut [M31C], twiddles: &[M31C]) {
    stockham_radix_8_dit_naive_non_8_last_impl(x, y, 1, x.len(), twiddles, true);
}

fn stockham_radix_8_dit_naive_non_8_first_impl(
    x: &mut [M31C],
    y: &mut [M31C],
    independent_fft_count: usize,
    independent_fft_len: usize,
    twiddles: &[M31C],
    eo: bool,
) {
    let sub_dft_len = if independent_fft_count == 1 {
        // is there a prettier way to express this?
        let rem = x.len().trailing_zeros() % 3;
        if rem == 0 {
            8
        } else {
            1 << rem
        }
    } else {
        8
    };
    let sub_dft_regions = x.len() / sub_dft_len / independent_fft_count;
    let sub_region_len = sub_dft_len * independent_fft_count;

    if independent_fft_len > 8 {
        stockham_radix_8_dit_naive_non_8_first_impl(
            y,
            x,
            independent_fft_count * sub_dft_len,
            independent_fft_len / sub_dft_len,
            twiddles,
            !eo,
        );

        // apply twiddles
        let len_over_8 = independent_fft_len / sub_dft_len;
        let shift = x.len() / independent_fft_len;
        for j in 1..len_over_8 {
            for i in 1..sub_dft_len {
                let twiddle_i_j = twiddles[i * j * shift];
                for k in 0..independent_fft_count {
                    x[k + independent_fft_count * (i + sub_dft_len * j)].mul_assign(&twiddle_i_j);
                }
            }
        }
    }

    // radix <= 8 exchanges
    let mut scratch = [M31C::ZERO; 8]; // always 8, must be constant
    for i in 0..sub_dft_regions {
        for j in 0..independent_fft_count {
            if !eo && independent_fft_len == 8 {
                for k in 0..sub_dft_len {
                    scratch[k] = y[j + independent_fft_count * k + sub_region_len * i];
                }
            } else {
                for k in 0..sub_dft_len {
                    scratch[k] = x[j + independent_fft_count * k + sub_region_len * i];
                }
            }
            match sub_dft_len {
                8 => size_8_radix_2_dit(&mut scratch),
                4 => size_4_radix_2_dit(&mut scratch),
                2 => size_2_radix_2_dit(&mut scratch),
                _ => (),
            }
            if independent_fft_len == 8 {
                for k in 0..sub_dft_len {
                    y[j + independent_fft_count * k + sub_region_len * i] = scratch[k];
                }
            } else {
                for k in 0..sub_dft_len {
                    x[j + independent_fft_count * k + sub_region_len * i] = scratch[k];
                }
            }
        }
    }

    if independent_fft_len == 8 {
        return;
    }

    // transpose within independent ffts
    let len_over_sub_len = independent_fft_len / sub_dft_len;
    for j in 0..len_over_sub_len {
        for i in 0..sub_dft_len {
            for k in 0..independent_fft_count {
                y[k + independent_fft_count * (j + len_over_sub_len * i)] =
                    x[k + independent_fft_count * (i + sub_dft_len * j)];
            }
        }
    }
}

pub fn stockham_radix_8_dit_naive_non_8_first(x: &mut [M31C], y: &mut [M31C], twiddles: &[M31C]) {
    stockham_radix_8_dit_naive_non_8_first_impl(x, y, 1, x.len(), twiddles, true);
}

#[test]
fn test_compare() {
    use crate::fft::bitreverse::bitreverse_enumeration_inplace;
    use crate::fft::fft::{rand_from_rng, serial_ct_ntt_natural_to_bitreversed};
    use crate::fft::tooling::{
        distribute_powers, domain_generator_for_size, precompute_twiddles_for_fft,
    };
    use std::time::{Duration, Instant};

    let worker = Worker::new();
    let mut rng = rand::thread_rng();

    fn flush(x: &[M31C], y: &mut [M31C]) -> Duration {
        let start = Instant::now();
        for i in 0..x.len() {
            y[i] = x[i];
        }
        start.elapsed()
    }

    fn do_one<F>(
        log_n: u32,
        input: &[M31C],
        reference: &[M31C],
        variant: &str,
        mut f: F,
    ) -> Duration
    where
        F: FnMut(&mut [M31C], &mut [M31C]) -> (),
    {
        let mut x = input.to_vec();
        let mut y = vec![M31C::ZERO; 1 << log_n];
        let start = Instant::now();
        f(&mut x, &mut y);
        let duration = start.elapsed();
        for (i, (output, control)) in y.iter().zip(reference).enumerate() {
            assert_eq!(
                output, control,
                "log_n = {}, {} failed at {}",
                log_n, variant, i
            );
        }
        duration
    }

    for log_n in 1..22 {
        let n = 1 << log_n;

        let mut input: Vec<M31C> = (0..n).map(|_| rand_from_rng(&mut rng)).collect();

        let x_flush = input.clone();
        let mut y_flush = input.clone();

        let twiddles = precompute_twiddles_for_fft::<M31C, false>(input.len(), &worker);
        let mut reference = input.clone();
        let start = Instant::now();
        serial_ct_ntt_natural_to_bitreversed(&mut reference, log_n, &twiddles[..]);
        let duration_reference = start.elapsed();
        let start = Instant::now();
        bitreverse_enumeration_inplace(&mut reference);
        let duration_bitrev = start.elapsed();

        let duration_flush_0 = flush(&x_flush, &mut y_flush);
        let duration_flush_1 = flush(&x_flush, &mut y_flush);

        let mut twiddles = vec![M31C::ONE; n];
        let omega = domain_generator_for_size::<M31C>(n as u64);
        distribute_powers(&mut twiddles, omega);

        flush(&x_flush, &mut y_flush);

        let duration_dif_cache_blocked =
            do_one(log_n, &input, &reference, "dif blocked", |x, y| {
                stockham_radix_8_dif_cache_blocked(x, y, &twiddles)
            });

        flush(&x_flush, &mut y_flush);

        let duration_dif_naive = do_one(log_n, &input, &reference, "dif naive", |x, y| {
            stockham_radix_8_dif_naive(x, y, &twiddles)
        });

        flush(&x_flush, &mut y_flush);

        let duration_dit_naive_non_8_last =
            do_one(log_n, &input, &reference, "dit naive non 8 last", |x, y| {
                stockham_radix_8_dit_naive_non_8_last(x, y, &twiddles)
            });

        flush(&x_flush, &mut y_flush);

        let duration_dit_naive_non_8_first = do_one(
            log_n,
            &input,
            &reference,
            "dit naive non 8 first",
            |x, y| stockham_radix_8_dit_naive_non_8_first(x, y, &twiddles),
        );

        let passes = (log_n + 7) / 3;
        let bandwidth_bound_estimate_0 = passes * duration_flush_0;
        let bandwidth_bound_estimate_1 = passes * duration_flush_1;

        println!("log_n = {:2} ", log_n);
        println!("    reference             {:?}", duration_reference);
        println!("    dif naive             {:?}", duration_dif_naive);
        println!("    dif blocked           {:?}", duration_dif_cache_blocked);
        println!(
            "    dit naive non 8 last  {:?}",
            duration_dit_naive_non_8_last
        );
        println!(
            "    dit naive non 8 first {:?}",
            duration_dit_naive_non_8_first
        );
        println!("    bitrev      {:?}", duration_bitrev);
        println!(
            "    bw bound estimates {:?} {:?}",
            bandwidth_bound_estimate_0, bandwidth_bound_estimate_1
        );
    }
}
