import re

'''
Removes puncuation/numeric/special/extra whitespace from original 'captions.txt'
Adds <start> and <end> tags used later for tokenizer

Example usage:
    labels = pd.read_csv('flickr8k/captions.txt')
    labels['caption'] = labels['caption'].apply(preprocess_text)
    labels.to_csv('flickr8k/captions_clean.csv', index=False)
'''
def preprocess_text(text):
    text = re.sub(r'[^a-z ]+', '', text.lower())
    text = '<start> ' + " ".join([word for word in text.split()]) + ' <end>'

    return text
