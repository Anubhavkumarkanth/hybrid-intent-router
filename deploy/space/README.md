---
title: Hybrid Intent Router
emoji: 🔀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: English-trained intent model + Gemini for unsure messages
---

# Hybrid Intent Router

Type a voice-bot style message in English, Hindi or Kannada and see which of 60 intents it is.

A small MuRIL model, fine-tuned on English data only, answers when it is confident. When it isn't, the
message goes to Gemini. The response shows the intent, the confidence, and which model answered.

The server is the same stack as the project: NGINX in front of gunicorn running a Flask app, on a free CPU.
Gemini calls are capped per minute; over the cap, unsure messages get the small model's answer
(`small_fallback`).

Code, results and how it was built: https://github.com/Anubhavkumarkanth/hybrid-intent-router
