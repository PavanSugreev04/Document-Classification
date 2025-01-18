from bertopic import BERTopic

TOPIC_NAMES = ["Graphics/outlier", "Technology", "Medical", "Sports",
               "Politics", "Graphics", "Space", "Entertainment",
               "Historical/War", "Food", "History/Egypt"]

def get_topic_prediction(input_text, model):
   
   topic_id, probability = model.transform(input_text)

   topic_name = TOPIC_NAMES[topic_id[0]+1]
   confidence = probability[0] * 100

   result = {
      "category": topic_name,
      "confidence": confidence
   }

   return result

classifier_model = BERTopic.load("my_11topicmodel")

def get_prediction(document_text):
   
   classification_result = get_topic_prediction(document_text, classifier_model)
   print(classification_result)
   return classification_result