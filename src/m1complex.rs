
use std::fmt::{Display, Formatter};

use crate::{field::{Field, FieldExtension, TwoAdicField}, m1::Mersenn31Field};

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Default)]
pub struct Mersenn31Complex {
    pub real_part: Mersenn31Field,
    pub imag_part: Mersenn31Field,
}

impl Mersenn31Complex {
    pub const fn new(real: Mersenn31Field, imag: Mersenn31Field) -> Self {
        Self { real_part: real, imag_part: imag, }
    }
    pub fn conjugate(&'_ mut self) -> &'_ mut Self {
        self.imag_part.negate();
        self
    }
    pub fn new_from_real(real: Mersenn31Field) -> Self {
        Self { real_part: real, imag_part: Mersenn31Field::ZERO, }
    }

    fn mul_naive(&'_ mut self, other: &Self) -> Self {
        let mut tmp = self.real_part;
        let mut real_chank_1 = tmp.mul_assign(&other.real_part);
        let mut tmp = self.imag_part;
        let real_chank_2 = tmp.mul_assign(&other.imag_part);
        let mut real = real_chank_1.sub_assign(&real_chank_2);
        let mut binding = self.real_part;
        let tmp = binding.mul_assign(&other.imag_part);
        let mut binding = self.imag_part;
        let tmp2 = binding.mul_assign(&other.real_part);
        let mut imag =  tmp.add_assign(&tmp2);
        Self::new(*real, *imag)
    }
    fn exp_power_of_2(&'_ mut self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res.mul_assign(&res.clone());
        }
        res
    }
    pub fn magnitude_squared(&self) -> Mersenn31Field {
        let mut left = self.real_part;
        let left = left.square();
        let mut right = self.imag_part;
        let right = right.square();
        let res = left.add_assign(right);
        *res
    }
    pub fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(
            self.real_part.div_2exp_u64(exp),
            self.imag_part.div_2exp_u64(exp),
        )
    }
    pub fn generator() -> Self {
        Self::new(Mersenn31Field::new(12), Mersenn31Field::ONE)
    }
}
impl TwoAdicField for Mersenn31Complex {
    const TWO_ADICITY: usize = 32;
    type BaseField = Mersenn31Field;

    fn two_adic_generator(bits: usize) -> Self {
        assert!(bits <= Self::TWO_ADICITY);
        let mut base = Self::new(
            Mersenn31Field::new(1_166_849_849),
            Mersenn31Field::new(1_117_296_306),
        );
        base.exp_power_of_2(Self::TWO_ADICITY - bits)
    }
}
impl Field for Mersenn31Complex {
    const ZERO: Self = Self::new(Mersenn31Field(0), Mersenn31Field(0));

    const ONE: Self = Self::new(Mersenn31Field(1), Mersenn31Field(0));

    const TWO: Self = Self::new(Mersenn31Field(2), Mersenn31Field(0));

    const MINUS_ONE: Self = Self::new(Mersenn31Field(Mersenn31Field::ORDER - 1), Mersenn31Field(0));


    fn as_u64(self) -> u64 {
        unreachable!()
    }

    fn from_u64_unchecked(value: u64) -> Self {
        unreachable!()
    }

    fn from_u64(value: u64) -> Option<Self> {
        unreachable!()
    }

    fn as_u64_reduced(&self) -> u64 {
        unreachable!()
    }

    fn is_zero(&self) -> bool {
        let flag = self.real_part.is_zero() && self.imag_part.is_zero();
        flag
    }

    fn as_boolean(&self) -> bool {
        unreachable!()
    }

    fn inverse(&self) -> Option<Self> {
        let mut tmp = *self;
        let a = self.magnitude_squared().inverse();
        let conj = tmp.conjugate();
        conj.mul_assign(&Self::new(a.unwrap(), Mersenn31Field::ZERO));
        let res = *conj;
        Some(res)
    }

    fn add_assign(&'_ mut self, other: &Self) -> &'_ mut Self {
        self.real_part.add_assign(&other.real_part);
        self.imag_part.add_assign(&other.imag_part);

        self
    }

    fn sub_assign(&'_ mut self, other: &Self) -> &'_ mut Self {
        self.real_part.sub_assign(&other.real_part);
        self.imag_part.sub_assign(&other.imag_part);
        self
    }

    fn mul_assign(&'_ mut self, other: &Self) -> &'_ mut Self {
        // Gauss algorithm for complex multiplication (a + ib) * (c + id)
        // t1 = a*c
        // t2 = b*d
        // re = t1 - t2
        // im = (a + b)*(c + d) - t1 - t2
        let mut a = self.real_part;
        let b = self.imag_part;
        let mut c = other.real_part;
        let d = other.imag_part;

        let mut t_1_res = self.real_part;
        t_1_res.mul_assign(&other.real_part);
        let t1_clone = t_1_res.clone();

        let mut t_2_res = self.imag_part;
        t_2_res.mul_assign(&other.imag_part);
        t_1_res.sub_assign(&t_2_res);

        a.add_assign(&b);
        c.add_assign(&d);

        a.mul_assign(&c);
        a.sub_assign(&t1_clone);
        a.sub_assign(&t_2_res);

        *self =  Self::new(t_1_res, a);
        self
    }

    fn square(&'_ mut self) -> &'_ mut Self {
        self.mul_assign(&self.clone())
    }

    fn negate(&'_ mut self) -> &'_ mut Self {
        todo!()
    }

    fn double(&'_ mut self) -> &'_ mut Self {
        todo!()
    }
    fn from_u32_with_reduction(c: u64) -> Self {
        let c = Mersenn31Field::from_nonreduced_u32(c);
        Self::new(c, Mersenn31Field::ZERO)
    }
}
impl Display for Mersenn31Complex {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} + {}i", self.real_part, self.imag_part)
    }
}


// use criterion::{black_box, criterion_group, criterion_main, Criterion};

// fn criterion_benchmark(c: &mut Criterion) {
//     // Type aliases for convenience
//     type Fi = Mersenn31Complex<Mersenn31Field>;
//     type F = Mersenn31Field;

//     // Preparing data for the first benchmark
//     let mut right_part = Fi::new(F::TWO, F::TWO);
//     let left_part = Fi::new(F::from_u64(4).unwrap(), F::from_u64(5).unwrap());

//     // First benchmark
//     c.bench_function("algorithm 1", |b| {
//         b.iter(|| right_part.mul_assign(black_box(&left_part)))
//     });

//     // Resetting data for the second benchmark
//     let mut right_part = Fi::new(F::TWO, F::TWO);
//     let left_part = Fi::new(F::from_u64(4).unwrap(), F::from_u64(5).unwrap());

//     // Second benchmark
//     c.bench_function("algorithm 2", |b| {
//         b.iter(|| right_part.mul_naive(black_box(&left_part)))
//     });
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul() {
        let mut binding = (Mersenn31Complex::new(Mersenn31Field::TWO, Mersenn31Field::TWO));
        let mut result = binding.mul_assign(&Mersenn31Complex::new(Mersenn31Field::from_u64(4).unwrap(), Mersenn31Field::from_u64(5).unwrap()));
        assert_eq!(
            *result,
            Mersenn31Complex::new(*Mersenn31Field::TWO.negate(), Mersenn31Field::from_u64(18).unwrap())

        );
    }
    #[test]
    fn test_add_assign() {
        // // Create two Mersenn31Complex instances
        // let mut complex1 = Mersenn31Complex::new(Mersenn31Field::new(1), Mersenn31Field::new(2));
        // let complex2 = Mersenn31Complex::new(Mersenn31Field::new(3), Mersenn31Field::new(4));

        // // Perform the addition
        // complex1.add_assign(&complex2);

        // // Expected result: (1 + 3) + (2 + 4)i = 4 + 6i
        // assert_eq!(complex1.real_part, Mersenn31Field::new(4));
        // assert_eq!(complex1.imag_part, Mersenn31Field::new(6));


        let mut complex1 = Mersenn31Complex::new(Mersenn31Field::new(1491731485), Mersenn31Field::new(864446369));
        let complex2 = Mersenn31Complex::new(Mersenn31Field::new(1491731485), Mersenn31Field::new(1283037278));

        // Perform the addition
        complex1.add_assign(&complex2);

        // Expected result: (1 + 3) + (2 + 4)i = 4 + 6i
        assert_eq!(complex1.real_part, Mersenn31Field::new(835979323));
        dbg!(complex1.imag_part);
        dbg!(Mersenn31Field::new(2147483647));
        assert_eq!(complex1.imag_part, Mersenn31Field::new(2147483647));


    }

}

