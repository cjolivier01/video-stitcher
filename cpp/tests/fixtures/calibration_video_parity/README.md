# Calibration video parity fixture

The stereo pair is six 640x360, 10 fps, lossless H.264 frames cropped from
one deterministic 960x360 source. The shared 320-pixel region therefore has
exact source correspondences. Regenerate it with FFmpeg 8 and libx264:

```bash
ffmpeg -y -f lavfi -i "nullsrc=size=960x360:rate=10:duration=1.2" \
  -vf "geq=lum='16+219*random(1)':cb=128:cr=128,boxblur=luma_radius=2:luma_power=2:chroma_radius=0:chroma_power=0,format=yuv420p" \
  -c:v rawvideo -pix_fmt yuv420p global.yuv

ffmpeg -y -f rawvideo -pixel_format yuv420p -video_size 960x360 -framerate 10 \
  -i global.yuv \
  -filter_complex "[0:v]split=2[l][r];[l]crop=640:360:0:0[left];[r]crop=640:360:320:0[right]" \
  -map "[left]" -frames:v 6 -c:v libx264 -preset veryslow -qp 0 -pix_fmt yuv420p \
  -g 6 -keyint_min 6 -sc_threshold 0 -bf 0 -threads 1 -movflags +faststart left.mp4 \
  -map "[right]" -frames:v 6 -c:v libx264 -preset veryslow -qp 0 -pix_fmt yuv420p \
  -g 6 -keyint_min 6 -sc_threshold 0 -bf 0 -threads 1 -movflags +faststart right.mp4
```

Fixture SHA-256 values:

```text
89c08be9318805e3951310de428f6fc942a0efeda0df2f605e24af8d2531bd37  left.mp4
662a43532a0ad9a0d9ddb255f27e4ed71efb8dbb997cdd156f2c88777f9294de  right.mp4
```
