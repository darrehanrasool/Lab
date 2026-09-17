import math

# QUESTION 1
# x² + 3x + 5

x1 = 2
answer1 = x1**2 + 3*x1 + 5
print("Q1 =", answer1)

# QUESTION 2
# (a + b) / (c - d)
a2 = 10
b2 = 5
c2 = 8
d2 = 3

answer2 = (a2 + b2) / (c2 - d2)
print("Q2 =", answer2)

# QUESTION 3
# √(x² + y²)

x3 = 3
y3 = 4

answer3 = math.sqrt(x3**2 + y3**2)
print("Q3 =", answer3)


# QUESTION 4
# (2x + 3y) / (x - y)

x4 = 5
y4 = 2

answer4 = (2*x4 + 3*y4) / (x4 - y4)
print("Q4 =", answer4)



# QUESTION 5
# e + (ln(x))^x
x5 = 2

answer5 = math.e + math.log(x5)**x5
print("Q5 =", answer5)


# QUESTION 6
# (x² + 2xy + y²) / (x² - y²)

x6 = 5
y6 = 2

answer6 = (x6**2 + 2*x6*y6 + y6**2) / (x6**2 - y6**2)
print("Q6 =", answer6)



# QUESTION 7
# √[(a² + b²) / (a - b)]

a7 = 5
b7 = 1

answer7 = math.sqrt((a7**2 + b7**2) / (a7 - b7))
print("Q7 =", answer7)



# QUESTION 8
# 1 / (1 + e^(-x))

x8 = 2

answer8 = 1 / (1 + math.exp(-x8))
print("Q8 =", answer8)



# QUESTION 9
# sin²(x) + cos²(x)

x9 = math.radians(45)

answer9 = math.sin(x9)**2 + math.cos(x9)**2
print("Q9 =", answer9)



# QUESTION 10
# [sin(x) + cos(x)] / [sin(x) - cos(x)]

x10 = math.radians(60)

answer10 = (math.sin(x10) + math.cos(x10)) / (math.sin(x10) - math.cos(x10))

print("Q10 =", answer10)



# QUESTION 15
# [(a+b)² - (a-b)²] / √(a²+b²)

a15 = 5
b15 = 3

answer15 = ((a15 + b15)**2 - (a15 - b15)**2) /  math.sqrt(a15**2 + b15**2)

print("Q15 =", answer15)