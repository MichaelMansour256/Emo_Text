from transformers import BertTokenizerFast, BertForSequenceClassification
class script():
    def __init__(self):
        self.model_path = "friends-bert-base-uncased"
        self.target_names = ["neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"]
        self.max_length = 512

        # reload our model/tokenizer. Optional, only usable when in Python files instead of notebooks
        self.model = BertForSequenceClassification.from_pretrained(self.model_path, num_labels=len(self.target_names))
        #.to("cuda")
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)


    def get_prediction(self,text):
        # prepare our text into tokenized sequence
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        #.to("cuda")
        # perform inference to our model
        outputs = self.model(**inputs)
        # get output probabilities by doing softmax
        probs = outputs[0].softmax(1)
        # executing argmax function to get the candidate label
        return self.target_names[probs.argmax()]


    #text = "i fall in love with you"
    #print(get_prediction(text))
