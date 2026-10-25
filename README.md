
# cal_hacks - using cartesia
# Argument Gym

**Argument Gym** is an adaptive AI debate sparring partner that helps students think critically, defend arguments, and respond to challenging questions under pressure.

The app also includes a speaking coach that analyzes speech delivery and tracks improvement over time.

## Setup

**Requirements:** Python 3.10+, a microphone, and Cartesia and Anthropic API keys.

1. Install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file:

   ```env
   CARTESIA_API_KEY=your_cartesia_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

4. Choose **AI Sparring** to practice debates or **Speaking Coach** to improve speech delivery. View saved reports under **History** and improvement charts under **Progress**.

## Cartesia Features Used

- **Ink Whisper Speech-to-Text:** Converts spoken responses into text using Cartesia's streaming WebSocket API.
- **Real-Time Audio Streaming:** Sends microphone audio as 16 kHz, 16-bit PCM chunks for transcription.
- **Transcript Finalization:** Retrieves speech transcripts throughout timed sessions.
- **Word-Level Transcription:** Uses recognized words and timestamps to help analyze speaking pace, pauses, and filler words.

Anthropic Claude processes transcripts to generate adaptive debate questions, evaluate arguments, and provide personalized feedback.

## Known Limitations

- **Microphone access:** Recording currently works locally; remote browser microphone support is not implemented.
- **Transcription accuracy:** Background noise, accents, and microphone quality may affect recognition.
- **AI feedback:** Argument scores are subjective estimates and may occasionally be inaccurate.
- **Vocal analysis:** Pitch and volume variation can be estimated, but emotional tone and confidence cannot be reliably determined from these measurements alone.
- **Privacy:** Audio is sent to Cartesia, and transcripts are sent to Claude. Session reports are stored locally; saving transcripts is optional.
- **User accounts:** Local profiles are not authenticated accounts, and cloud synchronization is not yet available.

## Future Improvements

Browser-based recording, live conversational AI debates, and more personalized exercises based on recurring argument weaknesses.
