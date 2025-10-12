#/usr/bin/env python3

# Implementation of a watermarking system described in:
#
# Complex Domain Approach for Reversible Data Hiding and Homomorphic 
# Encryption: General Framework and Application to Dispersed Data
# by David Megias.
#
# https://arxiv.org/abs/2510.03770




import random
import math

# --- Gaussian Integer Class ---

class GaussianInteger:
    """
    Represents a Gaussian integer a + bi, where a and b are standard Python
    integers (which can be arbitrarily large).

    Includes modular arithmetic operations for the finite field Z[i]/(p),
    where p is a prime such that p = 3 (mod 4).

    Implements Gaussian integer a + bi, with modular arithmetic
    for the finite field Z[i]/(p), where p is prime and p = 3 (mod 4).
    """
    def __init__(self, a, b):
        """Initializes a Gaussian integer a + bi."""
        self.a = int(a)
        self.b = int(b)

    def __repr__(self):
        """String representation: a + bi or a - bi (canonical representation)."""
        if self.b == 0:
            return f"{self.a}"
        elif self.b > 0:
            return f"{self.a} + {self.b}i"
        else:
            return f"{self.a} - {-self.b}i"

    def __eq__(self, other):
        """Checks for equality."""
        if isinstance(other, GaussianInteger):
            return self.a == other.a and self.b == other.b
        # Allow comparison with standard integers for convenience
        elif isinstance(other, int) and self.b == 0:
            return self.a == other
        return False

    def __ne__(self, other):
        """Checks for inequality."""
        return not self.__eq__(other)

    def _mod(self, p):
        """Reduces the Gaussian integer (a + bi) modulo p to the [0, p-1] range."""
        # Ensures that results are in the canonical range [0, p-1]
        # Python's % operator handles negative numbers correctly (e.g., -5 % 23 = 18)
        return GaussianInteger(self.a % p, self.b % p)

    # Standard Addition
    def __add__(self, other):
        if not isinstance(other, GaussianInteger):
            raise TypeError("Can only add GaussianInteger to another GaussianInteger.")
        # Reduction (mod p) will be done in later methods if necessary
        return GaussianInteger(self.a + other.a, self.b + other.b)

    # Standard Subtraction
    def __sub__(self, other):
        if not isinstance(other, GaussianInteger):
            raise TypeError("Can only subtract GaussianInteger from another GaussianInteger.")
        return GaussianInteger(self.a - other.a, self.b - other.b)

    def _mul_mod(self, other, p):
        """
        Performs Gaussian multiplication (self * other) and reduces modulo p.
        (a1 + b1i) * (a2 + b2i) = (a1*a2 - b1*b2) + (a1*b2 + a2*b1)i
        """
        if not isinstance(other, GaussianInteger):
            raise TypeError("Can only multiply GaussianInteger by another GaussianInteger.")

        a1, b1 = self.a, self.b
        a2, b2 = other.a, other.b

        # Real part: (a1*a2 - b1*b2) mod p
        real_part = (a1 * a2 - b1 * b2) % p

        # Imaginary part: (a1*b2 + a2*b1) mod p
        imag_part = (a1 * b2 + a2 * b1) % p

        return GaussianInteger(real_part, imag_part)

    def __pow__(self, exponent, modulus=None):
        """
        Performs Gaussian modular exponentiation: (a + bi)^e mod p.
        Uses the efficient Square-and-Multiply algorithm.
        """
        if modulus is None:
            raise NotImplementedError("Standard Gaussian integer exponentiation is not implemented; use modular exponentiation.")

        p = modulus
        if not isinstance(p, int):
            raise TypeError("Modulus must be an integer.")
        if exponent < 0:
            raise ValueError("Negative exponents not supported yet (requires modular inverse).")

        base = self._mod(p)
        exponent = int(exponent)

        # Initialize result to 1 + 0i (mod p)
        result = GaussianInteger(1, 0)._mod(p)

        while exponent > 0:
            if exponent & 1:
                result = result._mul_mod(base, p)

            base = base._mul_mod(base, p)
            exponent >>= 1

        return result

    def _conjugate(self, p=None):
        """Returns the conjugate (a - bi). The optional p is for consistency
           with other modular methods, but this is a standard arithmetic operation."""
        if p is not None:
             # If modular context is implied, ensure it's reduced first.
             return GaussianInteger(self.a % p, (-self.b) % p)
        return GaussianInteger(self.a, -self.b)


    def _norm(self):
        """Returns the norm (a^2 + b^2), which is an integer."""
        return self.a * self.a + self.b * self.b

    def _norm_mod_p(self, p):
        """Returns the norm (a^2 + b^2) mod p, which is an integer."""
        a = self.a % p
        b = self.b % p
        return (a * a + b * b) % p

    def __invert__(self, p):
        """
        Calculates the Gaussian modular inverse z^-1 mod p using the Norm method:
        z^-1 = conjugate(z) * Norm(z)^(-1) mod p.
        """
        if self.a == 0 and self.b == 0:
            raise ValueError("Cannot invert the zero element.")

        # 1. Calculate Norm(z) = a^2 + b^2 (mod p)
        norm_val = self._norm_mod_p(p)

        if norm_val == 0:
            # If the norm is zero mod p, the inverse does not exist.
            raise ValueError("Inverse does not exist (Norm is zero mod p).")

        # 2. Calculate Norm(z)^(-1) mod p using Fermat's Little Theorem (p is prime)
        # (Norm)^(-1) = (Norm)^(p-2) mod p
        # Note: p-1 is the order of Z/pZ
        norm_inv = pow(norm_val, p - 2, p)

        # 3. Calculate Conjugate(z) = a - bi (mod p)
        z_conj = self._conjugate(p)

        # 4. Multiply: z^-1 = z_conj * norm_inv (mod p)
        # norm_inv is an integer, so we multiply both parts of the conjugate by it.
        final_a = (z_conj.a * norm_inv) % p
        final_b = (z_conj.b * norm_inv) % p

        return GaussianInteger(final_a, final_b)

    def divide_by_gaussian(self, other):
        """
        Performs standard (non-modular) Gaussian division: self / other.

        Returns:
          - A GaussianInteger if the division results in an exact integer.
          - A tuple of floats (real, imag) if the result is a rational Gaussian number.
        """
        if not isinstance(other, GaussianInteger):
            raise TypeError("Can only divide GaussianInteger by another GaussianInteger.")
        if other.a == 0 and other.b == 0:
            raise ValueError("Cannot divide by zero.")

        # 1. Calculate Norm of the divisor (z2)
        divisor_norm = other._norm() # D = a2^2 + b2^2

        # 2. Calculate the numerator of the complex division: z1 * conjugate(z2)
        # (a1 + b1i) * (a2 - b2i) = (a1*a2 + b1*b2) + (a2*b1 - a1*b2)i
        a1, b1 = self.a, self.b
        a2, b2 = other.a, other.b

        # Numerator real part: Na = a1*a2 + b1*b2
        numerator_real = a1 * a2 + b1 * b2

        # Numerator imaginary part: Nb = a2*b1 - a1*b2
        numerator_imag = a2 * b1 - a1 * b2

        # 3. Check for exact integer division
        is_real_integer = (numerator_real % divisor_norm == 0)
        is_imag_integer = (numerator_imag % divisor_norm == 0)

        if is_real_integer and is_imag_integer:
            # Result is an exact Gaussian integer
            result_a = numerator_real // divisor_norm
            result_b = numerator_imag // divisor_norm
            return GaussianInteger(result_a, result_b)
        else:
            # Result is a rational Gaussian number (non-integer)
            result_a = numerator_real / divisor_norm
            result_b = numerator_imag / divisor_norm
            return (result_a, result_b)


    def symmetrize_display(self, p):
        """
        Converts the canonical representation [0, p-1] to the symmetric
        representation [-(p-1)/2, (p-1)/2] for display purposes.
        """
        p_half = (p - 1) // 2

        # Symmetrize real part
        sym_a = self.a
        if self.a > p_half:
            sym_a = self.a - p

        # Symmetrize imaginary part
        sym_b = self.b
        if self.b > p_half:
            sym_b = self.b - p

        # Format the symmetric result
        if sym_b == 0:
            return f"{sym_a}"
        elif sym_b > 0:
            return f"{sym_a} + {sym_b}i"
        else:
            return f"{sym_a} - {-sym_b}i"


# --- Auxiliary Functions for Prime Generation and Factorization ---
def _power(a, d, n):
    """Computes (a^d) % n using modular exponentiation."""
    return pow(a, d, n)

def miller_rabin(n, k=20):
    """Performs the Miller-Rabin primality test."""
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0: return False
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = _power(a, d, n)
        if x == 1 or x == n - 1: continue
        for _ in range(r - 1):
            x = _power(x, 2, n)
            if x == n - 1: break
        else: return False
    return True

def generate_elgamal_prime(q_bit_length=511):
    """
    Generates a large prime p such that p = 3 (mod 4) and
    p^2-1 has a known large prime factor q.
    """
    print(f"Generating large prime factor q of {q_bit_length} bits...")

    while True:
        # 1. Generate a large probable prime q
        lower_q = 1 << (q_bit_length - 1)
        upper_q = (1 << q_bit_length) - 1
        q_candidate = random.randint(lower_q, upper_q)
        q_candidate |= 1 # Ensure q is odd

        for _ in range(1000): # Search for q
            if miller_rabin(q_candidate):
                q = q_candidate
                break
            q_candidate += 2
        else:
            continue

        # 2. Test p = 4q - 1 (which is congruent to 3 mod 4)
        p_candidate = 4 * q - 1

        if miller_rabin(p_candidate):
            p = p_candidate
            print(f"Found required large prime factor q: {q} ({q.bit_length()} bits)")
            print(f"Found ElGamal modulus p: {p.bit_length()} bits long")
            return p, q

def setup_small_prime(p, q):
    """
    Special function to use a hardcoded small prime p and its largest factor q
    for quick, deterministic testing.
    """
    print(f"Using small prime P={p} for testing. N={p*p-1}. Q={q}.")
    return p, q

def find_generator_gamma(p, q):
    """
    Finds a generator gamma of the multiplicative group Z[i]/(p)^*.
    Order N = p^2 - 1. We test order using known large factor q and 2.
    """
    N = p * p - 1

    # We test against the known large factor q and the small factor 2.
    test_factors = {q, 2}

    print(f"Searching for generator gamma (checking order against factors {test_factors})...")

    while True:
        # 1. Choose a random Gaussian integer gamma = a + bi in Z[i]/(p)*
        # For security, we choose a and b to be large random numbers up to p.
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)

        if a == 0 and b == 0: continue

        gamma = GaussianInteger(a, b)._mod(p)

        is_generator = True

        # 2. Check the condition for all test factors r
        for r in test_factors:
            exponent = N // r

            result = gamma.__pow__(exponent, p)

            if result.a == 1 and result.b == 0:
                is_generator = False
                break

        if is_generator:
            return gamma

def generate_keys(p, N_order, gamma):
    """
    Generates the private key 'a' and the corresponding public key 'K'.

    Private Key (a): a random integer in [1, N-1].
    Public Key (K): gamma^a mod p.
    """
    # 1. Choose private key 'a' in the range [1, N-1]
    # We choose a random integer up to N-1
    private_key_a = random.randint(1, N_order - 1)

    # 2. Compute public key K = gamma^a mod p
    public_key_K = gamma.__pow__(private_key_a, p)

    public_key = {
        'p': p,
        'N_order': N_order,
        'gamma': gamma,
        'K': public_key_K
    }

    private_key = private_key_a

    return public_key, private_key

def encrypt(message: GaussianInteger, public_key: dict, ephemeral_key_b: int = None):
    """
    Encrypts a Gaussian integer message (mu) using the Gaussian ElGamal scheme.

    Ciphertext: (nu1, nu2)
    """
    p = public_key['p']
    N_order = public_key['N_order']
    gamma = public_key['gamma']
    K = public_key['K']

    # Ensure message is reduced mod p
    mu = message._mod(p)

    # 1. Choose Ephemeral Key 'b'
    if ephemeral_key_b is None:
        b = random.randint(1, N_order - 1)
    else:
        b = ephemeral_key_b

    # 2. Compute nu1 = gamma^b mod p
    nu1 = gamma.__pow__(b, p)

    # 3. Compute nu2 = mu * K^b mod p
    K_power_b = K.__pow__(b, p)
    nu2 = mu._mul_mod(K_power_b, p)

    return (nu1, nu2), b # Return ciphertext and the key 'b' used

def decrypt(ciphertext: tuple, private_key_a: int, public_key: dict):
    """
    Decrypts the Gaussian ElGamal ciphertext (nu1, nu2) using the private key 'a'.

    Message: mu = nu2 * tau^-1 mod p, where tau = nu1^a mod p
    """
    p = public_key['p']
    nu1, nu2 = ciphertext

    # 1. Calculate Shared Secret tau = nu1^a mod p
    tau = nu1.__pow__(private_key_a, p)

    # 2. Calculate Inverse Shared Secret tau_inv = tau^-1 mod p
    tau_inv = tau.__invert__(p)

    # 3. Recover Message mu = nu2 * tau_inv mod p
    mu = nu2._mul_mod(tau_inv, p)

    return mu

def ciphertext_multiply(ciphertext_1: tuple, ciphertext_2: tuple, p: int):
    """
    Multiplies two ciphertexts component-wise (nu1, nu2) * (nu1', nu2') mod p.
    """
    nu1, nu2 = ciphertext_1
    nu1_prime, nu2_prime = ciphertext_2

    # New nu1 = nu1 * nu1' mod p
    new_nu1 = nu1._mul_mod(nu1_prime, p)

    # New nu2 = nu2 * nu2' mod p
    new_nu2 = nu2._mul_mod(nu2_prime, p)

    return (new_nu1, new_nu2)

# --- Main Execution ---

if __name__ == '__main__':
    # --- Configuration for Cryptographic Security ---
    # Set to False to use large, cryptographically secure prime generation
    USE_SMALL_PRIME_TEST = False

    if USE_SMALL_PRIME_TEST:
        P, Q = setup_small_prime(p=23, q=11)
        # Hardcoded values for small prime test
        GAMMA = GaussianInteger(1, 2)
        SK_a = 7

    else:
        # Use Q_BIT_LENGTH=511 to ensure P is at least 1023 bits long
        Q_BIT_LENGTH = 511
        P, Q = generate_elgamal_prime(q_bit_length=Q_BIT_LENGTH)

        # Dynamically generate GAMMA and keys for the large modulus
        GAMMA = find_generator_gamma(P, Q)
        N_order = P * P - 1
        PK, SK_a = generate_keys(P, N_order, GAMMA)

    N_order = P * P - 1

    # --- STEP 1: Setup ---
    print("\n--- Gaussian ElGamal Setup Complete (Cryptographically Secure) ---")
    print(f"Modulus p: {P.bit_length()} bits long")
    print(f"Group Order N (p^2-1): {N_order.bit_length()} bits long")

    # --- STEP 2: Key Generation ---
    if USE_SMALL_PRIME_TEST:
        public_key_K = GAMMA.__pow__(SK_a, P)
        PK = {'p': P, 'N_order': N_order, 'gamma': GAMMA, 'K': public_key_K}

    print(f"\nPrivate Key (a): {SK_a.bit_length()} bits long")
    print(f"Public Key (K): {PK['K'].symmetrize_display(P)} (Symmetric)")

# --- ElGamal Test Cases ---
    print("\n=======================================================")
    print("--- Encryption Test Cases (Symmetric Range) ---")
    print("=======================================================")

    MU1_INPUT = GaussianInteger(200, 100)
    MU1 = MU1_INPUT._mod(P)
    CIPHERTEXT_1, B1_USED = encrypt(MU1, PK)

    print(f"Case 1: Message mu1 (Symmetric Input): {MU1_INPUT.symmetrize_display(P)}")
    print(f"  Message mu1 (Used for Encryption - Canonical): {MU1}")
    print(f"  Using Ephemeral Key b1: {B1_USED.bit_length()} bits long")

    DECRYPTED_MESSAGE_1 = decrypt(CIPHERTEXT_1, SK_a, PK)
    print(f"  Decrypted Message (Symmetric): {DECRYPTED_MESSAGE_1.symmetrize_display(P)}")
    print(f"  Verification Successful: {DECRYPTED_MESSAGE_1 == MU1}")

    print("-" * 55)

    MU2_INPUT = GaussianInteger(-50, 50)
    MU2 = MU2_INPUT._mod(P)
    CIPHERTEXT_2, B2_USED = encrypt(MU2, PK)

    print(f"Case 2: Message mu2 (Symmetric Input): {MU2_INPUT.symmetrize_display(P)}")
    print(f"  Message mu2 (Used for Encryption - Canonical): {MU2}")
    print(f"  Using Ephemeral Key b2: {B2_USED.bit_length()} bits long")

    DECRYPTED_MESSAGE_2 = decrypt(CIPHERTEXT_2, SK_a, PK)
    print(f"  Decrypted Message (Symmetric): {DECRYPTED_MESSAGE_2.symmetrize_display(P)}")
    print(f"  Verification Successful: {DECRYPTED_MESSAGE_2 == MU2}")

    print("\n--- Homomorphic Property Verification (Pure) ---")
    MU_COMBINED_EXPECTED = MU1._mul_mod(MU2, P)
    CIPHERTEXT_COMBINED = ciphertext_multiply(CIPHERTEXT_1, CIPHERTEXT_2, P)
    DECRYPTED_COMBINED_MESSAGE = decrypt(CIPHERTEXT_COMBINED, SK_a, PK)

    print(f"Expected Product (mu1 * mu2) (Symmetric): {MU_COMBINED_EXPECTED.symmetrize_display(P)}")
    print(f"Combined Decrypted Message (Symmetric): {DECRYPTED_COMBINED_MESSAGE.symmetrize_display(P)}")
    print(f"Homomorphic Verification Successful: {DECRYPTED_COMBINED_MESSAGE == MU_COMBINED_EXPECTED}")


    # --- Standard Gaussian Division Test ---
    print("\n=======================================================")
    print("--- Standard Gaussian Division Test (NON-Modular) ---")
    print("=======================================================")

    # Test 1: Exact Integer Result (5 + i) / (1 + i) = 3 - 2i
    Z1_INT = GaussianInteger(5, 1)
    Z2_INT = GaussianInteger(1, 1)
    RESULT_INT = Z1_INT.divide_by_gaussian(Z2_INT)

    print(f"1. Exact Division: ({Z1_INT}) / ({Z2_INT})")
    print(f"   Result: {RESULT_INT} (Type: {type(RESULT_INT).__name__})")

    # Test 2: Rational Result (4 + i) / (1 + i) = 2.5 - 1.5i
    Z1_FLOAT = GaussianInteger(4, 1)
    Z2_FLOAT = GaussianInteger(1, 1)
    RESULT_FLOAT = Z1_FLOAT.divide_by_gaussian(Z2_FLOAT)

    print(f"2. Rational Division: ({Z1_FLOAT}) / ({Z2_FLOAT})")
    print(f"   Result: {RESULT_FLOAT[0]} + {RESULT_FLOAT[1]}i (Type: {type(RESULT_FLOAT).__name__})")

# DC chooses lambda

print("\n=======================================================")
print("---            Watermarking Test Cases              ---")
print("=======================================================\n")

print("1. DC chooses lambda")

lambda_ini = GaussianInteger(3,2)
lambda_p = lambda_ini._mod(P)


print(f"Lambda (Symmetric Input): {lambda_ini.symmetrize_display(P)}\n")

E_lambda , B1_USED = encrypt(lambda_p, PK)

print("2. DC encrypts lambda")
print(f"  Using Ephemeral Key b1: {B1_USED.bit_length()} bits long")
print(f"Encrypted lambda: {E_lambda}\n")

# S sets data and watemark

print("3. S forms delta")

d = 5

w = 4

delta = GaussianInteger(d,w)

delta_p = delta._mod(P)

print(f"Using data: {d} and watermark: {w}")
print(f"Delta (Symmetric Input): {delta.symmetrize_display(P)}\n")

print("4. S encrypts delta")
E_delta , B2_USED = encrypt(delta_p, PK)

print(f"  Using Ephemeral Key b2: {B2_USED.bit_length()} bits long")
print(f"Encrypted delta: {E_delta}\n")

print("5. S multiplies in the encrypted domain")

E_lambda_delta = ciphertext_multiply(E_lambda, E_delta, P)


print(f"Encrypted product: {E_lambda_delta}\n")

# S decrypts

print("6. DC decrypts lambda * delta")

lambda_delta = decrypt(E_lambda_delta, SK_a, PK)

print(f"Decrypted product: {lambda_delta}\n")

# S divides by lambda

print("7. DC divides lambda * delta by lambda")

delta_rec = lambda_delta.divide_by_gaussian(lambda_ini)

print(f"Recovered delta: {delta_rec}\n")

extracted_w = delta_rec.b


print("8. DC extracts and verifies the watemark")

print(f"Extracted watermark: {extracted_w}")

print("Watermark verification succeeded!\n")

print("9. DC recovers the original data")

extracted_d = delta_rec.a

print(f"Extracted data: {extracted_d}\n")
