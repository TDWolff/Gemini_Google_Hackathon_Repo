from elevenlabs import Voice, VoiceSettings, play
from elevenlabs.client import ElevenLabs

client = ElevenLabs(
  api_key="aae657101a1f1cfa4cf53e2d78ca8338", # Defaults to ELEVEN_API_KEY
)

audio = client.generate(
    text="Hello! My name is Sigma male.",
    voice=Voice(
        voice_id='WjFqBUhW1Bg0nb8483pA',
        settings=VoiceSettings(stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True)
    )
)

play(audio)