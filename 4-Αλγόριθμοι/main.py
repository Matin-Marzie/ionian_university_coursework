# inf2022001 Mohammad Matin Marzie
# Αναβαθμολόγηση Μαθήματος

# ------------------------------Άσκηση-1------------------------------

# Υλοποιήθηκε με Αναβιβαστική bottom-up μέθοδο

def RodCutting(n, price):

    optimal_table = [0] * (n + 1)
    cuts = [0] * (n+1)

    for length in range(1, n + 1):

        for i in range(1, length + 1):
            tmp = price[i - 1] + optimal_table[length - i]
            if tmp > optimal_table[length]:
                optimal_table[length] = tmp
                cuts[length] = i

        # Πιο σύντομο με list Comprehensive και την συνάρτηση max() αλλά χάριν διευκολείας στην ανάγνωση γράφω και πιο αναλυτικά παραπάνω
        # optimal_table[length], cuts[length] = max(
        #     ((price[i - 1] + optimal_table[length - i], i) for i in range(1, length + 1)),
        #     key=lambda x: x[0]
        # )

    cut_needed = []
    while n > 0:
        cut_needed.append(cuts[n])
        n -= cuts[n]

    return optimal_table, cut_needed



# Runtime: Sum(1+2+...+n) = θ(n^2)
# Space  : Θ(n) για τον πίνακα optimal_table

if __name__ == '__main__':
    # μήκος ράβδου
    n=5
    # τιμές
    p = [2,5,7,8,10]

    best_price, cuts = RodCutting(n, p)
    print('Can sell with maximum profit of: ', best_price[n])
    print('Needed cuts: ', cuts)




# ------------------------------Άσκηση-4------------------------------
def RodCutting_1_3_4(n, price):
    optimal_table = [0] * (n + 1)
    cuts = [0] * (n + 1)
    allowed_lengths = [1, 3, 4]

    for length in range(1, n + 1):
        for i_allowed in allowed_lengths:
            if i_allowed <= length and i_allowed - 1 < len(price):
                tmp = price[i_allowed - 1] + optimal_table[length - i_allowed]
                if tmp > optimal_table[length]:
                    optimal_table[length] = tmp
                    cuts[length] = i_allowed

    cut_needed = []
    while n > 0:
        cut_needed.append(cuts[n])
        n -= cuts[n]

    return optimal_table, cut_needed

# Runtime: θ(n * k) όπου k = len(allowed_lengths)
# Space: Θ(n)

if __name__ == '__main__':
    # μήκος ράβδου
    n = 5
    # τιμές
    p = [2, 5, 7, 8, 10]

    best_price, cuts = RodCutting_1_3_4(n, p)
    print('Can sell with maximum profit of: ', best_price[n])
    print('Needed cuts: ', cuts)