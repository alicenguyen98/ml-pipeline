import sys
import numpy as np
import re

def try_parse(v: str, default=None):
	try:
		v = int(v)
		return v
	except:
		pass

	try:
		v = float(v)
		return v
	except:
		pass

	return default

def try_parse_array(v:str, default=None):
	try:
		# remove brackets
		if v[0] == '[' and v[-1] == ']':
			v = v[1:-1]
		v = np.fromstring(v, sep=',')
		return v
	except:
		pass

	return default


def transform_func_params(p):

	# Defines a recursive function
	def _transform(p):
		# Handle function expression
		if isinstance(p, str):
			# Check if this is a function expression (denoted by a custom prefix "func:" or "*func:")
			if re.match('\*?func:.+', p):
				v = transform_func_expression(p)		
				# Check whether this argument should be expanded (denoted by a leading *)
				expand = (isinstance(v, (list, tuple, np.ndarray))) and (p[0] == '*')
				return v, expand
		# Recursively transform every element in a list
		elif isinstance(p, list):
			i = 0
			while i < len(p):
				v, expand = _transform(p[i])
				# Should the argument be expanded?
				if expand:
					p.pop(i)
					p = [*p[:i], *v, *p[i:]]
					i += len(v)
				else:
					p[i] = v
					i += 1	
		# Recursively transform every element in a dict
		elif isinstance(p, dict):
			for k in p:
				p[k] = transform_func_params(p[k])
		
		return p, False

	return _transform(p)[0]
			
def transform_func_expression(p):

	try:
		# Slice expression		
		start = 6 if p[0] == '*' else 5
		fragments = p[start:].split(';')

		# parse function
		func_name = fragments[0].split('.')[-1]

		# parse module
		if len(fragments[0]) - 1 > len(func_name):
			module_name = fragments[0][:-len(func_name) - 1]

			# Only support numpy at the moment as this could be a security risk for malicious code
			if module_name not in ['numpy']:
				raise Exception(f"Unsupported module: {module_name}")

			module = sys.modules[module_name]
		else:
			raise Exception(f"Module not specified")
		
		func = getattr(module, func_name)
		if not callable(func):
			raise Exception(f"{func_name} not callable!")

		# parse arguments
		args = list()
		kwargs = dict()

		for arg in fragments[1:]:
			
			if not arg:
				continue

			def parse_arg_value(v):
				# Parse array string
				if v[0]=='[' and v[-1] ==']':
					parsed = try_parse_array(v)
					if not isinstance(parsed, type(None)):
						return parsed
				# Parse numeric value (int/float)
				parsed = try_parse(v)
				if not isinstance(parsed, type(None)):
					return parsed

				return v

			# Is this a named argument?
			if m := re.match('(.+)=(.+)', arg):
				# Handle keyword argument
				k, v = m.groups()
				kwargs[k] = parse_arg_value(v)
			else:
				# Handle positional argument
				if kwargs:
					raise Exception("Positional expression should be in front of named arguments!!") 
				args.append(parse_arg_value(arg))

		p = func(*args, **kwargs)

	except Exception as err:
		print(f"Failed to transform function expression for '{p}': {err}")

	return p