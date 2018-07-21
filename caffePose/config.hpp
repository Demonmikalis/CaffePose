#ifndef CONFIG_HPP
#define CONFIG_HPP

#define USE_OPENCV 1
#define CPU_ONLY 1
#define USE_CAFFE 1
#define UNIX_GUI 1


#define POSE_BODY_25_PAIRS_RENDER_GPU \
  1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   14,19,19,20,14,21, 11,22,22,23,11,24
#define POSE_BODY_25_SCALES_RENDER_GPU 1
#define POSE_BODY_25_COLORS_RENDER_GPU \
    255.f,     0.f,    85.f, \
    255.f,     0.f,     0.f, \
    255.f,    85.f,     0.f, \
    255.f,   170.f,     0.f, \
    255.f,   255.f,     0.f, \
    170.f,   255.f,     0.f, \
     85.f,   255.f,     0.f, \
      0.f,   255.f,     0.f, \
    255.f,     0.f,     0.f, \
      0.f,   255.f,    85.f, \
      0.f,   255.f,   170.f, \
      0.f,   255.f,   255.f, \
      0.f,   170.f,   255.f, \
      0.f,    85.f,   255.f, \
      0.f,     0.f,   255.f, \
    255.f,     0.f,   170.f, \
    170.f,     0.f,   255.f, \
    255.f,     0.f,   255.f, \
     85.f,     0.f,   255.f, \
      0.f,     0.f,   255.f, \
      0.f,     0.f,   255.f, \
      0.f,     0.f,   255.f, \
      0.f,   255.f,   255.f, \
      0.f,   255.f,   255.f, \
      0.f,   255.f,   255.f
#endif // CONFIG_HPP
