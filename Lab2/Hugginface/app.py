import gradio as gr
from pytube import YouTube
from transformers import pipeline
import os
from moviepy.editor import VideoFileClip


pipe = pipeline(model="GIanlucaRub/whisper-small-it-3",task="automatic-speech-recognition")

def transcribe_yt(link):
  yt = YouTube(link)
  audio = yt.streams.filter(only_audio=True)[0].download(filename="audio.mp3")
  text = pipe(audio)["text"]
  os.remove(audio)
  return text

def transcribe_audio(audio):
    text = pipe(audio)["text"]
    return text

def populate_metadata(link):
  yt = YouTube(link)
  return yt.thumbnail_url, yt.title

def transcribe_video(video):
    clip = VideoFileClip(video)
    audio = video[:-4] + ".mp3"
    clip.audio.write_audiofile(audio)
    clip.close()
    os.remove(video)
    text = transcribe_audio(audio)
    os.remove(audio)
    
    return text

block = gr.Blocks()

with block:
    gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <div>
                <h1 style="font-size: 400%;line-height: 1.2;">Whisper Italian Automatic Speech Recognition</h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 150%;margin-top: 30px;line-height: 1.2;">
                Realtime demo for Italian speech recognition using a fine-tuned Whisper Small model.You can use the model in 4 different ways.
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
          gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <p style="margin-bottom: 10px; font-size: 100%;margin-top: 10px;line-height: 1.2;">
                  Here you can see the transcription.
              </p>
            </div>
        """)
          text = gr.Textbox(
              label="Transcription", 
              placeholder="Transcription Output",
              lines=5)
          gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <p style="margin-bottom: 10px; font-size: 100%;margin-top: 20px;line-height: 1.0;">
                  You can record audio from your microphone.
              </p>
            </div>
        """)  
          microphone=gr.Audio(source="microphone", type="filepath")
          with gr.Row().style(mobile_collapse=False, equal_height=True): 
              btn_microphone = gr.Button("Transcribe microphone audio")
          
        
          gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <p style="margin-bottom: 10px; font-size: 100%;margin-top: 20px;line-height: 1.2;">
                  You can upload an audio file.
              </p>
            </div>
        """)  
          audio_uploaded=gr.Audio(source="upload", type="filepath")
          with gr.Row().style(mobile_collapse=False, equal_height=True): 
              btn_audio_uploaded = gr.Button("Transcribe audio uploaded")
          
        
        
          gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <p style="margin-bottom: 10px; font-size: 100%;margin-top: 20px;line-height: 1.2;">
                  You can upload a video file
              </p>
            </div>
        """) 
          video_uploaded = gr.Video(source = "upload") 
          with gr.Row().style(mobile_collapse=False, equal_height=True): 
              btn_video_uploaded = gr.Button("Transcribe video uploaded")
            
          
        
          gr.HTML(
        """

            <div style="text-align: center; max-width: 500px; margin: 0 auto;margin-top: 10px">
              <p style="margin-bottom: 10px; font-size: 100%;margin-top: 20px;line-height: 1.2;">
                  You can put a youtube video link
              </p>
            </div>
        """) 
          link = gr.Textbox(label="YouTube Link")
          with gr.Row().style(mobile_collapse=False, equal_height=True): 
              btn_youtube = gr.Button("Transcribe Youtube video") 
    
          with gr.Row().style(mobile_collapse=False, equal_height=True):
            title = gr.Label(label="Video Title", placeholder="Title")
            img = gr.Image(label="Thumbnail")
          
                
          
          # Events
          btn_youtube.click(transcribe_yt, inputs=[link], outputs=[text])
          btn_microphone.click(transcribe_audio, inputs=[microphone], outputs=[text])
          btn_audio_uploaded.click(transcribe_audio, inputs=[audio_uploaded], outputs=[text])
          btn_video_uploaded.click(transcribe_video, inputs=[video_uploaded], outputs=[text])
          link.change(populate_metadata, inputs=[link], outputs=[img, title])

block.launch(debug=True)