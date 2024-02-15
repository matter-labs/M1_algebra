
pub trait Field: 
    'static
    + Clone
    + Copy
    + std::fmt::Display
    + std::fmt::Debug
    + std::hash::Hash
    + std::cmp::PartialEq
    + std::cmp::Eq
    + std::marker::Send
    + std::marker::Sync
    + std::default::Default
{
    // const N = MersennePrime; // 4
    // identities
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const MINUS_ONE: Self;
    // const TWO_ADICITY: usize;

    fn as_u64(self) -> u64;
    fn from_u64_unchecked(value: u64) -> Self;
    fn from_u64(value: u64) -> Option<Self>;
    fn as_u64_reduced(&self) -> u64;
    // zero check
    fn is_zero(&self) -> bool;
    fn as_boolean(&self) -> bool; 
    
    fn from_boolean(flag: bool) -> Self {
        if flag { Self::ONE } else { Self::ZERO }
    }
    fn inverse(&self) -> Option<Self>;

    // add
    fn add_assign(&'_ mut self, other: &Self) -> &'_ mut Self;
    // sub
    fn sub_assign(&'_ mut self, other: &Self) -> &'_ mut Self;
    // mul
    fn mul_assign(&'_ mut self, other: &Self) -> &'_ mut Self;
    // square
    fn square(&'_ mut self) -> &'_ mut Self;
    // negate
    fn negate(&'_ mut self) -> &'_ mut Self;
    // double
    fn double(&'_ mut self) -> &'_ mut Self;
    fn pow_u64(&self, power: u64) -> Self {
        let mut current = *self;
        let mut product = Self::ONE;

        let mut j = 0;
        let num_bits = num_bits_u64(power);
        while j < num_bits {
            if (power >> j & 1) != 0 {
                product.mul_assign(&current);
            }
            current.square();
            j += 1;
        }

        product
    }
    #[inline(always)]
    fn from_u32_with_reduction(value: u64) -> Self;
}
pub trait FieldExtension<const DEGREE: usize>:
    'static
    + Clone
    + Copy
    + std::fmt::Display
    + std::fmt::Debug
    + std::hash::Hash
    + std::marker::Send
    + std::marker::Sync
{
    const TWO_ADICITY: usize;

    type BaseField: Field;
    // non-residue explicitly
    // fn non_residue() -> Self::BaseField;
    // generator's parametrization should also happen here
    // fn multiplicative_generator_coeffs() -> [Self::BaseField; DEGREE];
    // norm
    // fn compute_norm(el: &[Self::BaseField; DEGREE]) -> Self::BaseField;
    // there is no &self paramter here as we do not expect runtime parametrization
    // fn mul_by_non_residue(el: &mut Self::BaseField);
    fn two_adic_generator(bits: usize) -> Self;
}
pub trait TwoAdicField: Field {
    type BaseField: Field;
    /// The number of factors of two in this field's multiplicative group.
    const TWO_ADICITY: usize;

    /// Returns a generator of the multiplicative group of order `2^bits`.
    /// Assumes `bits < TWO_ADICITY`, otherwise the result is undefined.
    #[must_use]
    fn two_adic_generator(bits: usize) -> Self;
}
#[inline]
pub const fn num_bits_u64(n: u64) -> usize {
    (64 - n.leading_zeros()) as usize
}
