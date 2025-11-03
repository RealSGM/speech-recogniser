import os
from pydub import AudioSegment

def convert_m4a_to_wav(folder_path):
	for filename in os.listdir(folder_path):
		if filename.lower().endswith('.m4a'):
			m4a_path = os.path.join(folder_path, filename)
			wav_filename = os.path.splitext(filename)[0] + '.wav'
			wav_path = os.path.join(folder_path, wav_filename)

			print(f"Converting {filename} → {wav_filename} ...")
			sound = AudioSegment.from_file(m4a_path, format='m4a')
			sound.export(wav_path, format='wav')

	print("✅ Conversion complete!")


convert_m4a_to_wav("data/recordings/ahmed")
convert_m4a_to_wav("data/recordings/amber")
convert_m4a_to_wav("data/recordings/charlie")
convert_m4a_to_wav("data/recordings/christopher")
convert_m4a_to_wav("data/recordings/dominic")
convert_m4a_to_wav("data/recordings/emad")
convert_m4a_to_wav("data/recordings/emma")
convert_m4a_to_wav("data/recordings/hannah")
convert_m4a_to_wav("data/recordings/imogen")
convert_m4a_to_wav("data/recordings/jess")
convert_m4a_to_wav("data/recordings/josh")
convert_m4a_to_wav("data/recordings/joshua")
convert_m4a_to_wav("data/recordings/kailong")
convert_m4a_to_wav("data/recordings/kira")
convert_m4a_to_wav("data/recordings/manwel")