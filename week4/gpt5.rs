use std::time::Instant;

struct Lcg {
    state: u32,
}

impl Lcg {
    #[inline(always)]
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    #[inline(always)]
    fn next(&mut self) -> u32 {
        const A: u32 = 1_664_525;
        const C: u32 = 1_013_904_223;
        self.state = self.state.wrapping_mul(A).wrapping_add(C);
        self.state
    }
}

#[inline(always)]
fn fill_random_numbers(buf: &mut [i64], seed: u32, min_val: i64, max_val: i64) {
    let mut lcg = Lcg::new(seed);
    let range = (max_val - min_val + 1) as u64;
    for x in buf.iter_mut() {
        let v = lcg.next() as u64;
        let r = (v % range) as i64 + min_val;
        *x = r;
    }
}

#[inline(always)]
fn max_subarray_sum_bruteforce(nums: &[i64]) -> i64 {
    let n = nums.len();
    let mut max_sum = i64::MIN;

    for i in 0..n {
        let mut current_sum = 0i64;
        unsafe {
            let mut ptr = nums.as_ptr().add(i);
            let mut cnt = n - i;
            while cnt > 0 {
                current_sum += *ptr;
                if current_sum > max_sum {
                    max_sum = current_sum;
                }
                ptr = ptr.add(1);
                cnt -= 1;
            }
        }
    }
    max_sum
}

fn total_max_subarray_sum(n: usize, initial_seed: u32, min_val: i64, max_val: i64) -> i64 {
    let mut total_sum = 0i64;
    let mut seeder = Lcg::new(initial_seed);
    let mut buf = vec![0i64; n];

    for _ in 0..20 {
        let seed = seeder.next();
        fill_random_numbers(&mut buf, seed, min_val, max_val);
        total_sum += max_subarray_sum_bruteforce(&buf);
    }

    total_sum
}

fn main() {
    // Parameters
    let n: usize = 10_000;
    let initial_seed: u32 = 42;
    let min_val: i64 = -10;
    let max_val: i64 = 10;

    // Timing the function
    let start_time = Instant::now();
    let result = total_max_subarray_sum(n, initial_seed, min_val, max_val);
    let elapsed = start_time.elapsed().as_secs_f64();

    println!("Total Maximum Subarray Sum (20 runs): {}", result);
    println!("Execution Time: {:.6f} seconds", elapsed);
}