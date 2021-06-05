import pandas as pd 

model_names = [
	'densenet',
	'resnet',
	'inception',
	'effnet'
]

file_suffix = [
	'_img_features_{model}.csv'.format(model=model_name)
	for model_name in model_names
]

file_prefix = 'train'
merged = pd.read_csv(file_prefix + file_suffix[0])
for count, suffix in enumerate(file_suffix[1:]):
	filename = file_prefix + suffix
	print(filename)
	merged = merged.merge(
		pd.read_csv(filename), 
		on='Unnamed: 0', 
		validate='one_to_one',
		suffixes=('', '_{cnt}'.format(cnt=(count+1)))
		)
# Add class labels
merged = merged.merge(
	pd.read_csv('labels.csv').loc[
		:, ['image_name', 'target']
	],
	left_on='Unnamed: 0',
	right_on='image_name',
	validate='one_to_one'
)
merged.drop(columns=['Unnamed: 0'])
merged.to_csv('train.csv', index=False)

"""
merged = pd.read_csv('test1.csv')
for count, filename in enumerate(['test1.csv', 'test2.csv'][1:]):
	merged = merged.merge(
		pd.read_csv(filename), 
		on='Unnamed: 0', 
		validate='one_to_one',
		suffixes=('', '_{cnt}'.format(cnt=(count+1)))
		)
# Add class labels
merged = merged.merge(
	pd.read_csv('test_labels.csv').loc[
		:, ['image_name', 'target']
	],
	left_on='Unnamed: 0',
	right_on='image_name',
	validate='one_to_one'
)
merged.drop(columns=['Unnamed: 0'])
merged.to_csv('test3.csv', index=False)
print(merged)
"""


	

