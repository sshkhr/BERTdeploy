# BERTdeploy

This is a small app I built using HuggingFace Transformers and FastAPI to perform text classification using the pre-trained DistilBERT model. I mostly relied on the excellent tutorial by Venelin to build this (ref 1). I made a few key changes to his approach:  

* Used pre-trained model instead of fine-tuning
* Used `requirement.txt` for pip instead of using pipenv
* Did not use a lot of extra code style packages

**How to use?**

* `pip install requirements.txt`
* `bash bin/run_server`

Then make your API call:

```bash
http POST http://127.0.0.1:8000/classify text="Pre-trained DistilBERT seems to work quite well!"
```

You'll get an output like:


```js
{
    "confidence": 0.9998160004615784,
    "probabilities": {
        "negative": 0.00018407008610665798,
        "positive": 0.9998160004615784
    },
    "sentiment": "positive"
}
```

## TO-DO

* Deploy on Heroku - will need to find a workaround for downloading pre-trained models since Heroku's file system is ephemeral


## References

* [Deploy BERT for Sentiment Analysis as REST API using PyTorch, Transformers by Hugging Face and FastAPI](https://curiousily.com/posts/deploy-bert-for-sentiment-analysis-as-rest-api-using-pytorch-transformers-by-hugging-face-and-fastapi/)
* [Auto-deploy FastAPI App to Heroku via Git in these 5 Easy Steps](https://towardsdatascience.com/autodeploy-fastapi-app-to-heroku-via-git-in-these-5-easy-steps-8c7958ef5d41)