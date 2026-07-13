#!/bin/bash -ex

for text in "Great film, loved it!" "meh" "This was the worst movie I have ever seen in my entire life and I want my money back immediately" "ok"; do
  curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"$text\"}"
  echo
done
