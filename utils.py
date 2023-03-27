import re

'''
Removes puncuation/numeric/special/extra whitespace/single chars from original 'captions.txt'
Adds <start> and <end> tags used later for tokenizer
:param text: caption text string to process
:return: cleaned caption

Example usage:
    labels = pd.read_csv('flickr8k/captions.txt')
    labels['caption'] = labels['caption'].apply(preprocess_text)
    labels.to_csv('flickr8k/captions_clean.csv', index=False)
'''
def preprocess_text(text):
    text = re.sub(r'[^a-z ]+', '', text.lower())
    text = '<start> ' + " ".join([word for word in text.split() if len(word) > 1]) + ' <end>'

    return text
