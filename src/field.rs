
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
    // const N = MersinsePrime; // 4
    // identities
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const MINUS_ONE: Self;

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
}
