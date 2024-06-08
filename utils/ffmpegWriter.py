import cv2
import numpy as np
import subprocess as sp
import shlex

class FFMPEGWriter():
    def __init__(self, shape : np.ndarray, fps : float, output_filename = "output.mp4") -> None:
        self.width, self.height = shape
        self.output_filename = output_filename
        self.fps = fps
        self._is_closed = False
        self.process = sp.Popen(shlex.split(f'ffmpeg -loglevel quiet -stats -f rawvideo -pixel_format bgr24 -y -s {self.width}x{self.height} -hwaccel cuda -r {self.fps} -i - -c:v h264_nvenc "{self.output_filename}" '), stdin=sp.PIPE, stdout=sp.PIPE)
    def write(self, frame : cv2.Mat):
        frame = np.array(frame,dtype=np.uint8).tobytes()
        self.process.stdin.write(frame)
    def release(self):
        stdout, stderr = self.process.communicate()
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()
        self.process.stdin.close()
        self._is_closed = True
        if stderr:
            print(f"FFmpeg error: {stderr.decode()}")
            return None
        else:
            return stdout
    def __del__(self):
        if(not self._is_closed):
            self.release()
        