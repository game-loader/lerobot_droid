# LPWM native reference

Source: https://github.com/taldatech/lpwm
Revision: 4cf53c403433e64c01652ac2adbec66231a46dea
License: MIT (LICENSE alongside this file).

models.py, modules.py, vision_modules.py and loss_functions.py retain the complete
native DLP encoder/decoder, context posterior/prior, dynamics, sampling and loss
implementations. Only imports are rewritten to this local package. util_func.py
retains the unmodified computational helpers needed by these modules, removing
unused plotting, OpenCV, movie and training-log dependencies. Perceptual/VGG loss
remains optional and may download its native pretrained weights when explicitly
selected; the robot training path uses native pixel reconstruction.

The robot action-token Transformer is implemented separately in world_model.py;
it is intentionally NOT a claim of equivalence to the native context/AdaLN
dynamics. Full native architecture is accessible through native_reference().
