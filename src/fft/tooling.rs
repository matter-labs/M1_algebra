use crate::{field::TwoAdicField, worker::Worker, fft::bitreverse::bitreverse_enumeration_inplace};

pub(crate) fn distribute_powers<E: TwoAdicField>(input: &mut [E], element: E){
    let mut shift = E::ONE;
    let mut idx = 0;

    while idx < input.len() {
        input[idx].mul_assign(&shift);
        shift.mul_assign(&element);
        idx += 1;
    }
}

pub(crate) fn log2_n(n: usize) -> usize {
    debug_assert!(n.is_power_of_two());
    let res = n.trailing_zeros();
    res as usize
}

pub fn precompute_twiddles_for_fft<
    E: TwoAdicField,
    const INVERSED: bool,
>(
    fft_size: usize,
    worker: &Worker,
) -> Vec<E> {
    debug_assert!(fft_size.is_power_of_two());

    let mut omega = domain_generator_for_size::<E>(fft_size as u64);
    if INVERSED {
        omega = omega
            .inverse()
            .expect("must always exist for domain generator");
    }

    assert_eq!(omega.pow_u64(fft_size as u64), E::ONE);
    for i in 1..fft_size {
        assert_ne!(omega.pow_u64(i as u64), E::ONE);
    }

    let num_powers = fft_size / 2;
    let mut powers = materialize_powers_parallel(omega, num_powers, worker);
    bitreverse_enumeration_inplace(&mut powers);

    powers
}
pub fn domain_generator_for_size<E: TwoAdicField>(size: u64) -> E {
    debug_assert!(size.is_power_of_two());
    debug_assert!(size.trailing_zeros() as usize <= E::TWO_ADICITY);

    let mut omega = E::two_adic_generator(size.trailing_zeros() as usize);

    omega
}
pub(crate) fn materialize_powers_parallel<E: TwoAdicField>(
    base: E,
    size: usize,
    worker: &Worker,
) -> Vec<E> {
    if size == 0 {
        return Vec::new();
    }
    let mut storage = Vec::with_capacity(size);
    worker.scope(size, |scope, chunk_size| {
        for (chunk_idx, chunk) in storage.spare_capacity_mut()[..size]
            .chunks_mut(chunk_size)
            .enumerate()
        {
            scope.spawn(move |_| {
                let mut current = base.pow_u64((chunk_idx * chunk_size) as u64);
                for el in chunk.iter_mut() {
                    el.write(current);
                    current.mul_assign(&base);
                }
            });
        }
    });

    unsafe { storage.set_len(size) }

    storage
}