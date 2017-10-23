# Mean
def mean(values):
    return sum(values) / float(len(values))

# Variance
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])