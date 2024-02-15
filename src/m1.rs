/// The prime field `F_p` where `p = 2^31 - 1`.

use core::hash::Hash;
use core::fmt::Display;
use core::fmt::Formatter;
use core::fmt::Debug;
use core::hash::Hasher;
use std::fmt;
use std::ops::BitXorAssign;

use crate::field::Field;
use crate::field::FieldExtension;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mersenne31Field(pub u32);

impl Mersenne31Field{
    pub const ORDER: u32 = (1 << 31) - 1;
    pub const fn new(value: u32) -> Self{
        debug_assert!((value >> 31) == 0);
        
        Self(value)
    }
    #[inline(always)]
    const fn to_reduced_u32(&self) -> u32 {
        let mut c = self.0;
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        c
    }
    pub const fn from_nonreduced_u32(c: u64) -> Self {
        let mut c = c as u32;
        if c >= Self::ORDER {
            c -= Self::ORDER;
        }
        Self::new(c)
    }
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, multiplication by 2^k is just a left rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = (self.0 << exp) & ((1 << 31) - 1);
        let right = self.0 >> (31 - exp);
        let rotated = left | right;
        Self::new(rotated)
    }

    #[inline]
    pub fn div_2exp_u64(&self, exp: u64) -> Self {
        // In a Mersenne field, division by 2^k is just a right rotation by k bits.
        let exp = (exp % 31) as u8;
        let left = self.0 >> exp;
        let right = (self.0 << (31 - exp)) & ((1 << 31) - 1);
        let rotated = left | right;
        Self::new(rotated)
    }

    fn exp_power_of_2(&self, power_log: usize) -> Self {
        let mut res = *self;
        for _ in 0..power_log {
            res.square();
        }
        res
    }
    fn mod_pow(&self, mut exp: u32) -> Self {
        let mut base = *self;
        let mut result = &mut Mersenne31Field::new(1);
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul_assign(&base.clone());
            }
    
            exp >>= 1;
            base.mul_assign(&base.clone());
        }
    
        *result
    }
    // const fn compute_shifts() -> [Self; Self::CHAR_BITS] {
    //     let mut result = [Self::ZERO; Self::CHAR_BITS];
    //     let mut i = 0;
    //     while i < Self::CHAR_BITS {
    //         result[i] = Self::from_nonreduced_u32(1u64 << i);
    //         i += 1;
    //     }

    //     result
    // }
}
impl Default for Mersenne31Field {
    fn default() -> Self {
        Self(0u32)
    }
}

impl PartialEq for Mersenne31Field {
    fn eq(&self, other: &Self) -> bool {
        self.to_reduced_u32() == other.to_reduced_u32()
    }
}
impl Eq for Mersenne31Field {}

impl Hash for Mersenne31Field {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.to_reduced_u32())
    }
}

impl Ord for Mersenne31Field {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.to_reduced_u32().cmp(&other.to_reduced_u32())
    }
}

impl PartialOrd for Mersenne31Field {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Mersenne31Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl Debug for Mersenne31Field {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl Field for Mersenne31Field {
    const ZERO: Self = Self(0);
    const ONE: Self = Self(1);
    const TWO: Self = Self(2);
    const MINUS_ONE: Self = Self(Self::ORDER - 1);

    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.to_reduced_u32() == 0
    }


    #[inline(always)]
    fn as_u64(self) -> u64 {
        self.0 as u64
    }

    #[inline(always)]
    fn from_u64_unchecked(value: u64) -> Self{
        Self::new(
            value.try_into()
                .expect("Too large"),
        )
    }
    #[inline(always)]
    fn from_u64(value: u64) -> Option<Self> {
        if value as u32 >= Self::ORDER {
            None
        } else {
            Some(Self(value as u32))
        }
    }

    #[inline(always)]
    fn as_u64_reduced(&self) -> u64 {
        self.to_reduced_u32() as u64
    }

    fn as_boolean(&self) -> bool{
        let as_uint = self.to_reduced_u32();
        assert!(as_uint == 0 || as_uint == 1);
        as_uint != 0
    }

    fn inverse(&self) -> Option<Self>{
        //Since the nonzero elements of GF(pn) form a finite group with respect to multiplication,
        // a^p^n−1 = 1 (for a ≠ 0), thus the inverse of a is a^p^n−2.
        if self.is_zero() {
            return None;
        }
        Some(self.mod_pow(Mersenne31Field::ORDER - 2))
    }
    
    fn add_assign(&'_ mut self, other: &Self) -> &'_ mut Self{
        let mut sum = self.0.wrapping_add(other.0);
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum += u32::from(msb != 0);
        if sum >= Self::ORDER {
            sum -= Self::ORDER;
        }
        self.0 = sum;

        self
    }
    // sub
    fn sub_assign(&'_ mut self, other: &Self) -> &'_ mut Self{
        let mut sum = self.0.wrapping_sub(other.0);
        let msb = sum & (1 << 31);
        sum.bitxor_assign(msb);
        sum -= u32::from(msb != 0);
        self.0 = sum;

        self

    }
    // mul
    fn mul_assign(&'_ mut self, other: &Self) -> &'_ mut Self{
        let product = u64::from(self.0) * u64::from(other.0);
        let product_low = (product as u32) & ((1 << 31) - 1);
        let product_high = (product >> 31) as u32;
        *self = Self(product_low);
        self.add_assign(&Self(product_high));
        self
    }
    // square
    fn square(&'_ mut self) -> &'_ mut Self{
        self.mul_assign(&self.clone())

    }
    // negate
    #[inline(always)]
    fn negate(&'_ mut self) -> &'_ mut Self{
        if self.is_zero() == false {
            *self = Self(Self::ORDER - self.to_reduced_u32());
        }
    
        self
    }
    // double
    fn double(&'_ mut self) -> &'_ mut Self{
        let t = *self;
        self.add_assign(&t);

        self
    }
    #[inline(always)]
    fn from_u32_with_reduction(value: u64) -> Self {
        Self::from_nonreduced_u32(value)
    }

}
impl FieldExtension<2> for Mersenne31Field{
    const TWO_ADICITY: usize = 2;

    type BaseField = Mersenne31Field;

    fn two_adic_generator(bits: usize) -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn basic_properties() {
        assert_eq!(Mersenne31Field::ZERO, Mersenne31Field(0));
        assert_eq!(Mersenne31Field::ONE, Mersenne31Field(1));
        assert_eq!(Mersenne31Field::TWO, Mersenne31Field(2));
        assert_eq!(Mersenne31Field::MINUS_ONE, Mersenne31Field(Mersenne31Field::ORDER - 1));
    }

    #[test]
    fn test_multiplication() {
        let a = Mersenne31Field::from_u64(5).unwrap();
        let b = Mersenne31Field::from_u64(4).unwrap();
        let mut res = a;
        res.mul_assign(&b);
        assert_eq!(res.to_reduced_u32(), 20);

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 1).unwrap();
        let b = Mersenne31Field::from_u64(2).unwrap();
        let mut result = a;
        result.mul_assign(&b);
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap());

        let a = Mersenne31Field::from_u64(12345).unwrap();
        let b = Mersenne31Field::ONE;
        let mut result = a;
        result.mul_assign(&b);
        assert_eq!(result, Mersenne31Field::from_u64(12345).unwrap());

        let a = Mersenne31Field::from_u64(12345).unwrap();
        let b = Mersenne31Field::ZERO;
        let mut result = a;
        result.mul_assign(&b);
        assert_eq!(result, Mersenne31Field::ZERO);

        let a = Mersenne31Field::from_u64(17).unwrap();
        let b = Mersenne31Field::from_u64(19).unwrap();
        let mut result1 = a;
        let mut result2 = b;
        result1.mul_assign(&b);
        result2.mul_assign(&a);
        assert_eq!(result1, result2);

        let a = Mersenne31Field::from_u64(123).unwrap();
        let b = Mersenne31Field::from_u64(456).unwrap();
        let mut result = a;
        result.mul_assign(&b);
        assert_eq!(result, Mersenne31Field::from_u64(56088 % Mersenne31Field::ORDER as u64).unwrap());

    }

    #[test]
    fn test_square() {
        let mut a = Mersenne31Field::from_u64(7).unwrap();
        a.square();
        assert_eq!(a.to_reduced_u32(), 49);
        let a = Mersenne31Field::from_u64(2).unwrap();
        let mut result = a.clone();
        result.square();
        assert_eq!(result, Mersenne31Field::from_u64(4).unwrap());

        let a = Mersenne31Field::from_u64(3).unwrap();
        let mut result = a.clone();
        result.square();
        assert_eq!(result, Mersenne31Field::from_u64(9).unwrap());

        let a = Mersenne31Field::ZERO;
        let mut result = a.clone();
        result.square();
        assert_eq!(result, Mersenne31Field::ZERO);

        let a = Mersenne31Field::ONE;
        let mut result = a.clone();
        result.square();
        assert_eq!(result, Mersenne31Field::ONE);

    }

    #[test]
    fn test_negate() {
        let mut a = Mersenne31Field::from_u64(5).unwrap();
        a.negate();
        assert_eq!(a.to_reduced_u32(), Mersenne31Field::ORDER - 5);

        let a = Mersenne31Field::from_u64(2).unwrap();
        let mut result = a.clone();
        result.negate();
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap());

        let a = Mersenne31Field::from_u64(3).unwrap();
        let mut result = a.clone();
        result.negate();
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 3).unwrap());

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 1).unwrap();
        let mut result = a.clone();
        result.negate();
        assert_eq!(result, Mersenne31Field::ONE);

        let a = Mersenne31Field::ZERO;
        let mut result = a.clone();
        result.negate();
        assert_eq!(result, Mersenne31Field::ZERO);

        let a = Mersenne31Field::from_u64(12345).unwrap();
        let mut result = a.clone();
        result.negate().negate();
        assert_eq!(result, a);

        let a = Mersenne31Field::from_u64(123456).unwrap();
        let mut result = a.clone();
        result.negate();
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 123456).unwrap());
    }

    #[test]
    fn test_double() {
        let mut a = Mersenne31Field::from_u64(12345).unwrap();
        a.double();
        assert_eq!(a.to_reduced_u32(), 24690);

        let a = Mersenne31Field::from_u64(2).unwrap();
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::from_u64(4).unwrap());

        let a = Mersenne31Field::from_u64(3).unwrap();
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::from_u64(6).unwrap());

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap();
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 4).unwrap());

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 1).unwrap();
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap());

        let a = Mersenne31Field::ZERO;
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::ZERO);

        let a = Mersenne31Field::ONE;
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::TWO);

        let a = Mersenne31Field::from_u64(123456).unwrap();
        let mut result = a.clone();
        result.double();
        assert_eq!(result, Mersenne31Field::from_u64(246912).unwrap());
    }

    #[test]
    fn test_inverse() {
        let a = Mersenne31Field::from_u64(4).unwrap();
        let inv_a = a.inverse().unwrap();
        let mut res = a;
        res.mul_assign(&inv_a);
        // a * a^-1 should be 1
        assert_eq!(res.to_reduced_u32(), 1);

        let a = Mersenne31Field::from_u64(2).unwrap();
        let inv_a = a.inverse().unwrap();
        let mut res = a;
        res.mul_assign(&inv_a);
        assert_eq!(res, Mersenne31Field::ONE);

        let a = Mersenne31Field::from_u64(3).unwrap();
        let inv_a = a.inverse().unwrap();
        let mut res = a;
        res.mul_assign(&inv_a);
        assert_eq!(res.to_reduced_u32(), 1);

        let a = Mersenne31Field::ONE;
        let inv_a = a.inverse().unwrap();
        assert_eq!(inv_a, Mersenne31Field::ONE);

        let a = Mersenne31Field::ZERO;
        assert!(a.inverse().is_none());

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap();
        let inv_a = a.inverse().unwrap();
        let mut res = a;
        res.mul_assign(&inv_a);
        assert_eq!(res, Mersenne31Field::ONE);

        let a = Mersenne31Field::from_u64(4).unwrap();
        let inv_a = a.inverse().unwrap();
        let double_inv_a = inv_a.inverse().unwrap();
        assert_eq!(a, double_inv_a);

        let a = Mersenne31Field::from_u64(123456).unwrap();
        let inv_a = a.inverse().unwrap();
        let mut res = a;
        res.mul_assign(&inv_a);
        assert_eq!(res, Mersenne31Field::ONE);

    }

    #[test]
    fn test_add() {
        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap();
        let b = Mersenne31Field::from_u64(10).unwrap();
        let mut res = a;
        res.add_assign(&b);
        assert_eq!(res.to_reduced_u32(), 8);  
        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap();
        let b = Mersenne31Field::from_u64(4).unwrap();
        let mut sum = a;
        sum.add_assign(&b);
        assert_eq!(sum, Mersenne31Field::from_u64(2).unwrap());

        let a = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 1).unwrap();
        let b = Mersenne31Field::ONE;
        let mut sum = a;
        sum.add_assign(&b);
        assert_eq!(sum, Mersenne31Field::ZERO);

        let a = Mersenne31Field::from_u64(10).unwrap();
        let b = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 8).unwrap();
        let mut sum = a;
        sum.add_assign(&b);
        assert_eq!(sum, Mersenne31Field::from_u64(2).unwrap());

        let a = Mersenne31Field::from_u64(10).unwrap();
        let b = Mersenne31Field::from_u64(20).unwrap();
        let mut sum = a;
        sum.add_assign(&b);
        assert_eq!(sum, Mersenne31Field::from_u64(30).unwrap());
    }

    #[test]
    fn test_sub() {
        let a = Mersenne31Field::from_u64(2).unwrap();
        let b = Mersenne31Field::from_u64(10).unwrap();
        let mut res = a;
        res.sub_assign(&b);
        assert_eq!(res.to_reduced_u32(), Mersenne31Field::ORDER - 8);

        let a = Mersenne31Field::from_u64(3).unwrap();
        let b = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 2).unwrap();
        let mut result = a;
        result.sub_assign(&b);
        assert_eq!(result, Mersenne31Field::from_u64(5).unwrap());

        let min_val = 0; 
        let a = Mersenne31Field::from_u64(min_val as u64).unwrap();
        let b = Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 1).unwrap();
        let mut result = a;
        result.sub_assign(&b);
        assert_eq!(result, Mersenne31Field::ONE);

        let a = Mersenne31Field::from_u64(10).unwrap();
        let b = Mersenne31Field::from_u64(15).unwrap();
        let mut result = a;
        result.sub_assign(&b);
        assert_eq!(result, Mersenne31Field::from_u64(Mersenne31Field::ORDER as u64 - 5).unwrap());
    }
    #[test]
    fn test_count(){
        let num = Mersenne31Field::from_u64(5).unwrap();
        let a = num.exp_power_of_2(2);
    }
}