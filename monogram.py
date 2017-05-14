import subprocess

def generate_monogram(monogram_text):
    monogram = None
    try:
      monogram = subprocess.check_output(
          [
              "./font_text/text-to-svg", 
              monogram_text
            ])
    except subprocess.CalledProcessError as e:
      print(str(e))
    return monogram