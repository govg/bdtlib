import os
import sys

def derivative_real(y, y_exp, x, factor):
	return (y - y_exp)*x / factor

def derivative_binary(y, y_exp, x, factor):
	# if y_exp >= 0.5:
	# 	y_pred = 1
	# else:
	# 	y_pred = 0

	# if y == y_pred:
	# 	der = 0*x
	# else:
	# 	der = y_exp*(1-y_exp)*x  / factor

	der = (y - y_exp)*x / factor

	return der