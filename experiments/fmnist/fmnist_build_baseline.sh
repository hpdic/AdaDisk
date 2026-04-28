#!/bin/bash
set -e

DISKANN_HOME=${HOME}/hpdic/AdaDisk
BUILDER_BIN=${DISKANN_HOME}/build/apps/build_disk_index
RAW_DATA=${HOME}/hpdic/data_fmnist/fashion_base.bin
OUTPUT_DIR=${HOME}/hpdic/data_fmnist/indices

R_VAL=32
L_VAL=50

RAM_BUDGET=2
THREADS=96

INDEX_NAME=diskann_base_R${R_VAL}_L${L_VAL}_B${RAM_BUDGET}G
INDEX_PREFIX=${OUTPUT_DIR}/${INDEX_NAME}

mkdir -p ${OUTPUT_DIR}
rm -f ${INDEX_PREFIX}*

echo 'Start building fmnist baseline...'

${BUILDER_BIN} \
    --data_type float \
    --dist_fn l2 \
    --data_path ${RAW_DATA} \
    --index_path_prefix ${INDEX_PREFIX} \
    -R ${R_VAL} \
    -L ${L_VAL} \
    -B ${RAM_BUDGET} \
    -M ${RAM_BUDGET} \
    -T ${THREADS}

# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$ bash fmnist_build_baseline.sh 
# Start building fmnist baseline...

# [HPDIC MOD: Apr 23 2026, 15:23:13]

# Starting index build: R=32 L=50 Query RAM budget: 1.87905e+09 Indexing ram budget: 2 T: 96
# Compressing 784-dimensional data into 512 bytes per vector.
# Opened: /home/cc/hpdic/data_fmnist/fashion_base.bin, size: 188160008, cache_size: 67108864
# Training data with 60000 samples loaded.
# Processing chunk 0 with dimensions [0, 2)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 1 with dimensions [2, 4)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 2 with dimensions [4, 6)
# Residuals unchanged: 89.2287 becomes 89.2287. Early termination.
# Processing chunk 3 with dimensions [6, 8)
# Residuals unchanged: 12514.6 becomes 12514.6. Early termination.
# Processing chunk 4 with dimensions [8, 10)
# Processing chunk 5 with dimensions [10, 12)
# Processing chunk 6 with dimensions [12, 14)
# Processing chunk 7 with dimensions [14, 16)
# Processing chunk 8 with dimensions [16, 18)
# Processing chunk 9 with dimensions [18, 20)
# Processing chunk 10 with dimensions [20, 22)
# Processing chunk 11 with dimensions [22, 24)
# Residuals unchanged: 7709.35 becomes 7709.35. Early termination.
# Processing chunk 12 with dimensions [24, 26)
# Residuals unchanged: 323.109 becomes 323.109. Early termination.
# Processing chunk 13 with dimensions [26, 28)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 14 with dimensions [28, 30)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 15 with dimensions [30, 32)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 16 with dimensions [32, 34)
# Residuals unchanged: 8230.62 becomes 8230.62. Early termination.
# Processing chunk 17 with dimensions [34, 36)
# Processing chunk 18 with dimensions [36, 38)
# Processing chunk 19 with dimensions [38, 40)
# Processing chunk 20 with dimensions [40, 42)
# Processing chunk 21 with dimensions [42, 44)
# Processing chunk 22 with dimensions [44, 46)
# Processing chunk 23 with dimensions [46, 48)
# Processing chunk 24 with dimensions [48, 50)
# Processing chunk 25 with dimensions [50, 52)
# Residuals unchanged: 64963.9 becomes 64963.9. Early termination.
# Processing chunk 26 with dimensions [52, 54)
# Residuals unchanged: 15165.2 becomes 15165.2. Early termination.
# Processing chunk 27 with dimensions [54, 56)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 28 with dimensions [56, 58)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 29 with dimensions [58, 60)
# Residuals unchanged: 188.154 becomes 188.154. Early termination.
# Processing chunk 30 with dimensions [60, 62)
# Residuals unchanged: 55171.6 becomes 55171.6. Early termination.
# Processing chunk 31 with dimensions [62, 64)
# Processing chunk 32 with dimensions [64, 66)
# Processing chunk 33 with dimensions [66, 68)
# Processing chunk 34 with dimensions [68, 70)
# Processing chunk 35 with dimensions [70, 72)
# Processing chunk 36 with dimensions [72, 74)
# Processing chunk 37 with dimensions [74, 76)
# Processing chunk 38 with dimensions [76, 78)
# Processing chunk 39 with dimensions [78, 80)
# Processing chunk 40 with dimensions [80, 82)
# Residuals unchanged: 43061.9 becomes 43061.9. Early termination.
# Processing chunk 41 with dimensions [82, 84)
# Residuals unchanged: 751.187 becomes 751.187. Early termination.
# Processing chunk 42 with dimensions [84, 86)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 43 with dimensions [86, 88)
# Residuals unchanged: 4082.86 becomes 4082.86. Early termination.
# Processing chunk 44 with dimensions [88, 90)
# Processing chunk 45 with dimensions [90, 92)
# Processing chunk 46 with dimensions [92, 94)
# Processing chunk 47 with dimensions [94, 96)
# Processing chunk 48 with dimensions [96, 98)
# Processing chunk 49 with dimensions [98, 100)
# Processing chunk 50 with dimensions [100, 102)
# Processing chunk 51 with dimensions [102, 104)
# Processing chunk 52 with dimensions [104, 106)
# Processing chunk 53 with dimensions [106, 108)
# Processing chunk 54 with dimensions [108, 110)
# Residuals unchanged: 85605.9 becomes 85605.9. Early termination.
# Processing chunk 55 with dimensions [110, 112)
# Residuals unchanged: 6598.95 becomes 6598.95. Early termination.
# Processing chunk 56 with dimensions [112, 114)
# Residuals unchanged: 17.6 becomes 17.6. Early termination.
# Processing chunk 57 with dimensions [114, 116)
# Residuals unchanged: 18469.1 becomes 18469.1. Early termination.
# Processing chunk 58 with dimensions [116, 118)
# Processing chunk 59 with dimensions [118, 120)
# Processing chunk 60 with dimensions [120, 122)
# Processing chunk 61 with dimensions [122, 124)
# Processing chunk 62 with dimensions [124, 126)
# Processing chunk 63 with dimensions [126, 128)
# Processing chunk 64 with dimensions [128, 130)
# Processing chunk 65 with dimensions [130, 132)
# Processing chunk 66 with dimensions [132, 134)
# Processing chunk 67 with dimensions [134, 136)
# Processing chunk 68 with dimensions [136, 138)
# Processing chunk 69 with dimensions [138, 140)
# Residuals unchanged: 20093.8 becomes 20093.8. Early termination.
# Processing chunk 70 with dimensions [140, 142)
# Residuals unchanged: 1147.45 becomes 1147.45. Early termination.
# Processing chunk 71 with dimensions [142, 144)
# Residuals unchanged: 44903.9 becomes 44903.9. Early termination.
# Processing chunk 72 with dimensions [144, 146)
# Processing chunk 73 with dimensions [146, 148)
# Processing chunk 74 with dimensions [148, 150)
# Processing chunk 75 with dimensions [150, 152)
# Processing chunk 76 with dimensions [152, 154)
# Processing chunk 77 with dimensions [154, 156)
# Processing chunk 78 with dimensions [156, 158)
# Processing chunk 79 with dimensions [158, 160)
# Processing chunk 80 with dimensions [160, 162)
# Processing chunk 81 with dimensions [162, 164)
# Processing chunk 82 with dimensions [164, 166)
# Processing chunk 83 with dimensions [166, 168)
# Residuals unchanged: 42360.1 becomes 42360.1. Early termination.
# Processing chunk 84 with dimensions [168, 170)
# Residuals unchanged: 4968.51 becomes 4968.51. Early termination.
# Processing chunk 85 with dimensions [170, 172)
# Residuals unchanged: 77069.9 becomes 77069.9. Early termination.
# Processing chunk 86 with dimensions [172, 174)
# Processing chunk 87 with dimensions [174, 176)
# Processing chunk 88 with dimensions [176, 178)
# Processing chunk 89 with dimensions [178, 180)
# Processing chunk 90 with dimensions [180, 182)
# Processing chunk 91 with dimensions [182, 184)
# Processing chunk 92 with dimensions [184, 186)
# Processing chunk 93 with dimensions [186, 188)
# Processing chunk 94 with dimensions [188, 190)
# Processing chunk 95 with dimensions [190, 192)
# Processing chunk 96 with dimensions [192, 194)
# Processing chunk 97 with dimensions [194, 196)
# Residuals unchanged: 60988.8 becomes 60988.8. Early termination.
# Processing chunk 98 with dimensions [196, 198)
# Residuals unchanged: 12065.3 becomes 12065.3. Early termination.
# Processing chunk 99 with dimensions [198, 200)
# Processing chunk 100 with dimensions [200, 202)
# Processing chunk 101 with dimensions [202, 204)
# Processing chunk 102 with dimensions [204, 206)
# Processing chunk 103 with dimensions [206, 208)
# Processing chunk 104 with dimensions [208, 210)
# Processing chunk 105 with dimensions [210, 212)
# Processing chunk 106 with dimensions [212, 214)
# Processing chunk 107 with dimensions [214, 216)
# Processing chunk 108 with dimensions [216, 218)
# Processing chunk 109 with dimensions [218, 220)
# Processing chunk 110 with dimensions [220, 222)
# Processing chunk 111 with dimensions [222, 224)
# Processing chunk 112 with dimensions [224, 226)
# Residuals unchanged: 19197.2 becomes 19197.2. Early termination.
# Processing chunk 113 with dimensions [226, 228)
# Processing chunk 114 with dimensions [228, 230)
# Processing chunk 115 with dimensions [230, 232)
# Processing chunk 116 with dimensions [232, 234)
# Processing chunk 117 with dimensions [234, 236)
# Processing chunk 118 with dimensions [236, 238)
# Processing chunk 119 with dimensions [238, 240)
# Processing chunk 120 with dimensions [240, 242)
# Processing chunk 121 with dimensions [242, 244)
# Processing chunk 122 with dimensions [244, 246)
# Processing chunk 123 with dimensions [246, 248)
# Processing chunk 124 with dimensions [248, 250)
# Processing chunk 125 with dimensions [250, 252)
# Processing chunk 126 with dimensions [252, 254)
# Residuals unchanged: 26345.5 becomes 26345.5. Early termination.
# Processing chunk 127 with dimensions [254, 256)
# Processing chunk 128 with dimensions [256, 258)
# Processing chunk 129 with dimensions [258, 260)
# Processing chunk 130 with dimensions [260, 262)
# Processing chunk 131 with dimensions [262, 264)
# Processing chunk 132 with dimensions [264, 266)
# Processing chunk 133 with dimensions [266, 268)
# Processing chunk 134 with dimensions [268, 270)
# Processing chunk 135 with dimensions [270, 272)
# Processing chunk 136 with dimensions [272, 274)
# Processing chunk 137 with dimensions [274, 276)
# Processing chunk 138 with dimensions [276, 278)
# Processing chunk 139 with dimensions [278, 280)
# Processing chunk 140 with dimensions [280, 282)
# Residuals unchanged: 34338 becomes 34338. Early termination.
# Processing chunk 141 with dimensions [282, 284)
# Processing chunk 142 with dimensions [284, 286)
# Processing chunk 143 with dimensions [286, 288)
# Processing chunk 144 with dimensions [288, 290)
# Processing chunk 145 with dimensions [290, 292)
# Processing chunk 146 with dimensions [292, 294)
# Processing chunk 147 with dimensions [294, 296)
# Processing chunk 148 with dimensions [296, 298)
# Processing chunk 149 with dimensions [298, 300)
# Processing chunk 150 with dimensions [300, 302)
# Processing chunk 151 with dimensions [302, 304)
# Processing chunk 152 with dimensions [304, 306)
# Processing chunk 153 with dimensions [306, 308)
# Processing chunk 154 with dimensions [308, 310)
# Residuals unchanged: 42139 becomes 42139. Early termination.
# Processing chunk 155 with dimensions [310, 312)
# Processing chunk 156 with dimensions [312, 314)
# Processing chunk 157 with dimensions [314, 316)
# Processing chunk 158 with dimensions [316, 318)
# Processing chunk 159 with dimensions [318, 320)
# Processing chunk 160 with dimensions [320, 322)
# Processing chunk 161 with dimensions [322, 324)
# Processing chunk 162 with dimensions [324, 326)
# Processing chunk 163 with dimensions [326, 328)
# Processing chunk 164 with dimensions [328, 330)
# Processing chunk 165 with dimensions [330, 332)
# Processing chunk 166 with dimensions [332, 334)
# Processing chunk 167 with dimensions [334, 336)
# Processing chunk 168 with dimensions [336, 338)
# Residuals unchanged: 55248.8 becomes 55248.8. Early termination.
# Processing chunk 169 with dimensions [338, 340)
# Processing chunk 170 with dimensions [340, 342)
# Processing chunk 171 with dimensions [342, 344)
# Processing chunk 172 with dimensions [344, 346)
# Processing chunk 173 with dimensions [346, 348)
# Processing chunk 174 with dimensions [348, 350)
# Processing chunk 175 with dimensions [350, 352)
# Processing chunk 176 with dimensions [352, 354)
# Processing chunk 177 with dimensions [354, 356)
# Processing chunk 178 with dimensions [356, 358)
# Processing chunk 179 with dimensions [358, 360)
# Processing chunk 180 with dimensions [360, 362)
# Processing chunk 181 with dimensions [362, 364)
# Processing chunk 182 with dimensions [364, 366)
# Residuals unchanged: 67269.4 becomes 67268.7. Early termination.
# Processing chunk 183 with dimensions [366, 368)
# Processing chunk 184 with dimensions [368, 370)
# Processing chunk 185 with dimensions [370, 372)
# Processing chunk 186 with dimensions [372, 374)
# Processing chunk 187 with dimensions [374, 376)
# Processing chunk 188 with dimensions [376, 378)
# Processing chunk 189 with dimensions [378, 380)
# Processing chunk 190 with dimensions [380, 382)
# Processing chunk 191 with dimensions [382, 384)
# Processing chunk 192 with dimensions [384, 386)
# Processing chunk 193 with dimensions [386, 388)
# Processing chunk 194 with dimensions [388, 390)
# Processing chunk 195 with dimensions [390, 392)
# Processing chunk 196 with dimensions [392, 394)
# Processing chunk 197 with dimensions [394, 396)
# Processing chunk 198 with dimensions [396, 398)
# Processing chunk 199 with dimensions [398, 400)
# Processing chunk 200 with dimensions [400, 402)
# Processing chunk 201 with dimensions [402, 404)
# Processing chunk 202 with dimensions [404, 406)
# Processing chunk 203 with dimensions [406, 408)
# Processing chunk 204 with dimensions [408, 410)
# Processing chunk 205 with dimensions [410, 412)
# Processing chunk 206 with dimensions [412, 414)
# Processing chunk 207 with dimensions [414, 416)
# Processing chunk 208 with dimensions [416, 418)
# Processing chunk 209 with dimensions [418, 420)
# Processing chunk 210 with dimensions [420, 422)
# Processing chunk 211 with dimensions [422, 424)
# Processing chunk 212 with dimensions [424, 426)
# Processing chunk 213 with dimensions [426, 428)
# Processing chunk 214 with dimensions [428, 430)
# Processing chunk 215 with dimensions [430, 432)
# Processing chunk 216 with dimensions [432, 434)
# Processing chunk 217 with dimensions [434, 436)
# Processing chunk 218 with dimensions [436, 438)
# Processing chunk 219 with dimensions [438, 440)
# Processing chunk 220 with dimensions [440, 442)
# Processing chunk 221 with dimensions [442, 444)
# Processing chunk 222 with dimensions [444, 446)
# Processing chunk 223 with dimensions [446, 448)
# Processing chunk 224 with dimensions [448, 450)
# Processing chunk 225 with dimensions [450, 452)
# Processing chunk 226 with dimensions [452, 454)
# Processing chunk 227 with dimensions [454, 456)
# Processing chunk 228 with dimensions [456, 458)
# Processing chunk 229 with dimensions [458, 460)
# Processing chunk 230 with dimensions [460, 462)
# Processing chunk 231 with dimensions [462, 464)
# Processing chunk 232 with dimensions [464, 466)
# Processing chunk 233 with dimensions [466, 468)
# Processing chunk 234 with dimensions [468, 470)
# Processing chunk 235 with dimensions [470, 472)
# Processing chunk 236 with dimensions [472, 474)
# Processing chunk 237 with dimensions [474, 476)
# Processing chunk 238 with dimensions [476, 478)
# Processing chunk 239 with dimensions [478, 480)
# Processing chunk 240 with dimensions [480, 482)
# Processing chunk 241 with dimensions [482, 484)
# Processing chunk 242 with dimensions [484, 486)
# Processing chunk 243 with dimensions [486, 488)
# Processing chunk 244 with dimensions [488, 490)
# Processing chunk 245 with dimensions [490, 492)
# Processing chunk 246 with dimensions [492, 494)
# Processing chunk 247 with dimensions [494, 496)
# Processing chunk 248 with dimensions [496, 498)
# Processing chunk 249 with dimensions [498, 500)
# Processing chunk 250 with dimensions [500, 502)
# Processing chunk 251 with dimensions [502, 504)
# Processing chunk 252 with dimensions [504, 506)
# Processing chunk 253 with dimensions [506, 508)
# Processing chunk 254 with dimensions [508, 510)
# Processing chunk 255 with dimensions [510, 512)
# Processing chunk 256 with dimensions [512, 514)
# Processing chunk 257 with dimensions [514, 516)
# Processing chunk 258 with dimensions [516, 518)
# Processing chunk 259 with dimensions [518, 520)
# Processing chunk 260 with dimensions [520, 522)
# Processing chunk 261 with dimensions [522, 524)
# Processing chunk 262 with dimensions [524, 526)
# Processing chunk 263 with dimensions [526, 528)
# Processing chunk 264 with dimensions [528, 530)
# Processing chunk 265 with dimensions [530, 532)
# Processing chunk 266 with dimensions [532, 534)
# Processing chunk 267 with dimensions [534, 536)
# Processing chunk 268 with dimensions [536, 538)
# Processing chunk 269 with dimensions [538, 540)
# Processing chunk 270 with dimensions [540, 542)
# Processing chunk 271 with dimensions [542, 544)
# Processing chunk 272 with dimensions [544, 545)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 273 with dimensions [545, 546)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 274 with dimensions [546, 547)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 275 with dimensions [547, 548)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 276 with dimensions [548, 549)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 277 with dimensions [549, 550)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 278 with dimensions [550, 551)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 279 with dimensions [551, 552)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 280 with dimensions [552, 553)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 281 with dimensions [553, 554)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 282 with dimensions [554, 555)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 283 with dimensions [555, 556)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 284 with dimensions [556, 557)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 285 with dimensions [557, 558)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 286 with dimensions [558, 559)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 287 with dimensions [559, 560)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 288 with dimensions [560, 561)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 289 with dimensions [561, 562)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 290 with dimensions [562, 563)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 291 with dimensions [563, 564)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 292 with dimensions [564, 565)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 293 with dimensions [565, 566)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 294 with dimensions [566, 567)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 295 with dimensions [567, 568)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 296 with dimensions [568, 569)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 297 with dimensions [569, 570)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 298 with dimensions [570, 571)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 299 with dimensions [571, 572)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 300 with dimensions [572, 573)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 301 with dimensions [573, 574)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 302 with dimensions [574, 575)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 303 with dimensions [575, 576)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 304 with dimensions [576, 577)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 305 with dimensions [577, 578)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 306 with dimensions [578, 579)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 307 with dimensions [579, 580)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 308 with dimensions [580, 581)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 309 with dimensions [581, 582)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 310 with dimensions [582, 583)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 311 with dimensions [583, 584)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 312 with dimensions [584, 585)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 313 with dimensions [585, 586)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 314 with dimensions [586, 587)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 315 with dimensions [587, 588)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 316 with dimensions [588, 589)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 317 with dimensions [589, 590)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 318 with dimensions [590, 591)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 319 with dimensions [591, 592)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 320 with dimensions [592, 593)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 321 with dimensions [593, 594)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 322 with dimensions [594, 595)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 323 with dimensions [595, 596)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 324 with dimensions [596, 597)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 325 with dimensions [597, 598)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 326 with dimensions [598, 599)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 327 with dimensions [599, 600)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 328 with dimensions [600, 601)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 329 with dimensions [601, 602)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 330 with dimensions [602, 603)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 331 with dimensions [603, 604)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 332 with dimensions [604, 605)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 333 with dimensions [605, 606)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 334 with dimensions [606, 607)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 335 with dimensions [607, 608)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 336 with dimensions [608, 609)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 337 with dimensions [609, 610)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 338 with dimensions [610, 611)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 339 with dimensions [611, 612)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 340 with dimensions [612, 613)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 341 with dimensions [613, 614)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 342 with dimensions [614, 615)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 343 with dimensions [615, 616)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 344 with dimensions [616, 617)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 345 with dimensions [617, 618)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 346 with dimensions [618, 619)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 347 with dimensions [619, 620)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 348 with dimensions [620, 621)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 349 with dimensions [621, 622)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 350 with dimensions [622, 623)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 351 with dimensions [623, 624)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 352 with dimensions [624, 625)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 353 with dimensions [625, 626)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 354 with dimensions [626, 627)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 355 with dimensions [627, 628)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 356 with dimensions [628, 629)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 357 with dimensions [629, 630)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 358 with dimensions [630, 631)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 359 with dimensions [631, 632)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 360 with dimensions [632, 633)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 361 with dimensions [633, 634)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 362 with dimensions [634, 635)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 363 with dimensions [635, 636)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 364 with dimensions [636, 637)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 365 with dimensions [637, 638)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 366 with dimensions [638, 639)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 367 with dimensions [639, 640)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 368 with dimensions [640, 641)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 369 with dimensions [641, 642)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 370 with dimensions [642, 643)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 371 with dimensions [643, 644)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 372 with dimensions [644, 645)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 373 with dimensions [645, 646)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 374 with dimensions [646, 647)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 375 with dimensions [647, 648)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 376 with dimensions [648, 649)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 377 with dimensions [649, 650)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 378 with dimensions [650, 651)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 379 with dimensions [651, 652)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 380 with dimensions [652, 653)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 381 with dimensions [653, 654)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 382 with dimensions [654, 655)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 383 with dimensions [655, 656)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 384 with dimensions [656, 657)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 385 with dimensions [657, 658)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 386 with dimensions [658, 659)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 387 with dimensions [659, 660)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 388 with dimensions [660, 661)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 389 with dimensions [661, 662)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 390 with dimensions [662, 663)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 391 with dimensions [663, 664)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 392 with dimensions [664, 665)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 393 with dimensions [665, 666)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 394 with dimensions [666, 667)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 395 with dimensions [667, 668)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 396 with dimensions [668, 669)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 397 with dimensions [669, 670)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 398 with dimensions [670, 671)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 399 with dimensions [671, 672)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 400 with dimensions [672, 673)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 401 with dimensions [673, 674)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 402 with dimensions [674, 675)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 403 with dimensions [675, 676)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 404 with dimensions [676, 677)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 405 with dimensions [677, 678)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 406 with dimensions [678, 679)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 407 with dimensions [679, 680)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 408 with dimensions [680, 681)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 409 with dimensions [681, 682)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 410 with dimensions [682, 683)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 411 with dimensions [683, 684)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 412 with dimensions [684, 685)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 413 with dimensions [685, 686)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 414 with dimensions [686, 687)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 415 with dimensions [687, 688)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 416 with dimensions [688, 689)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 417 with dimensions [689, 690)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 418 with dimensions [690, 691)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 419 with dimensions [691, 692)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 420 with dimensions [692, 693)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 421 with dimensions [693, 694)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 422 with dimensions [694, 695)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 423 with dimensions [695, 696)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 424 with dimensions [696, 697)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 425 with dimensions [697, 698)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 426 with dimensions [698, 699)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 427 with dimensions [699, 700)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 428 with dimensions [700, 701)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 429 with dimensions [701, 702)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 430 with dimensions [702, 703)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 431 with dimensions [703, 704)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 432 with dimensions [704, 705)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 433 with dimensions [705, 706)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 434 with dimensions [706, 707)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 435 with dimensions [707, 708)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 436 with dimensions [708, 709)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 437 with dimensions [709, 710)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 438 with dimensions [710, 711)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 439 with dimensions [711, 712)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 440 with dimensions [712, 713)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 441 with dimensions [713, 714)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 442 with dimensions [714, 715)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 443 with dimensions [715, 716)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 444 with dimensions [716, 717)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 445 with dimensions [717, 718)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 446 with dimensions [718, 719)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 447 with dimensions [719, 720)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 448 with dimensions [720, 721)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 449 with dimensions [721, 722)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 450 with dimensions [722, 723)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 451 with dimensions [723, 724)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 452 with dimensions [724, 725)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 453 with dimensions [725, 726)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 454 with dimensions [726, 727)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 455 with dimensions [727, 728)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 456 with dimensions [728, 729)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 457 with dimensions [729, 730)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 458 with dimensions [730, 731)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 459 with dimensions [731, 732)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 460 with dimensions [732, 733)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 461 with dimensions [733, 734)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 462 with dimensions [734, 735)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 463 with dimensions [735, 736)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 464 with dimensions [736, 737)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 465 with dimensions [737, 738)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 466 with dimensions [738, 739)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 467 with dimensions [739, 740)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 468 with dimensions [740, 741)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 469 with dimensions [741, 742)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 470 with dimensions [742, 743)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 471 with dimensions [743, 744)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 472 with dimensions [744, 745)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 473 with dimensions [745, 746)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 474 with dimensions [746, 747)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 475 with dimensions [747, 748)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 476 with dimensions [748, 749)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 477 with dimensions [749, 750)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 478 with dimensions [750, 751)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 479 with dimensions [751, 752)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 480 with dimensions [752, 753)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 481 with dimensions [753, 754)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 482 with dimensions [754, 755)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 483 with dimensions [755, 756)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 484 with dimensions [756, 757)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 485 with dimensions [757, 758)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 486 with dimensions [758, 759)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 487 with dimensions [759, 760)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 488 with dimensions [760, 761)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 489 with dimensions [761, 762)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 490 with dimensions [762, 763)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 491 with dimensions [763, 764)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 492 with dimensions [764, 765)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 493 with dimensions [765, 766)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 494 with dimensions [766, 767)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 495 with dimensions [767, 768)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 496 with dimensions [768, 769)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 497 with dimensions [769, 770)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 498 with dimensions [770, 771)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 499 with dimensions [771, 772)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 500 with dimensions [772, 773)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 501 with dimensions [773, 774)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 502 with dimensions [774, 775)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 503 with dimensions [775, 776)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 504 with dimensions [776, 777)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 505 with dimensions [777, 778)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 506 with dimensions [778, 779)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 507 with dimensions [779, 780)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 508 with dimensions [780, 781)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 509 with dimensions [781, 782)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 510 with dimensions [782, 783)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Processing chunk 511 with dimensions [783, 784)
# Residuals unchanged: 3.40282e+38 becomes 0. Early termination.
# Writing bin: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin
# bin: #pts = 256, #dims = 784, size = 802824B
# Finished writing bin.
# Writing bin: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin
# bin: #pts = 784, #dims = 1, size = 3144B
# Finished writing bin.
# Writing bin: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin
# bin: #pts = 513, #dims = 1, size = 2060B
# Finished writing bin.
# Writing bin: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin
# bin: #pts = 4, #dims = 1, size = 40B
# Finished writing bin.
# Saved pq pivot data to /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin of size 812124B.
# Opened: /home/cc/hpdic/data_fmnist/fashion_base.bin, size: 188160008, cache_size: 67108864
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 4, #dims = 1...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 256, #dims = 784...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 784, #dims = 1...
# done.
# Reading bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin ...
# Opening bin file /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_pq_pivots.bin... 
# Metadata: #pts = 513, #dims = 1...
# done.
# Loaded PQ pivot information
# Processing points  [0, 60000)...done.
# Time for generating quantized data: 1325.319824 seconds
# Full index fits in RAM budget, should consume at most 0.20594GiBs, so building in one shot
# L2: Using AVX2 distance computation DistanceL2Float
# Passed, empty search_params while creating index config
# Using only first 60000 from file.. 
# Starting index build with 60000 points... 

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>> [HPDIC] SUCCESS: You have reached the CORE logic! <<
# >>> Function: Index::link()                           <<
# >>> Status:   Ready for theoretical modifications     <<
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# >>> [MCGI] Module Loaded Successfully.
# 0% of index build completed.Starting final cleanup..done. Link time: 9.85983s
# Index built with degree: max:32  avg:32  min:32  count(deg<2):0
# Not saving tags as they are not enabled.
# Time taken for save: 0.318083s.
# Time for building merged vamana index: 10.333436 seconds
# Opened: /home/cc/hpdic/data_fmnist/fashion_base.bin, size: 188160008, cache_size: 67108864
# Vamana index file size=7920024
# Opened: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_disk.index, cache_size: 67108864
# medoid: 37961B
# max_node_len: 3268B
# nnodes_per_sector: 1B
# # sectors: 60000
# Sector #0written
# Finished writing 245764096B
# Writing bin: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_disk.index
# bin: #pts = 9, #dims = 1, size = 80B
# Finished writing bin.
# Output disk index file written to /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_disk.index
# Finished writing 245764096B
# Time for generating disk layout: 0.403375 seconds
# Opened: /home/cc/hpdic/data_fmnist/fashion_base.bin, size: 188160008, cache_size: 67108864
# Loading base /home/cc/hpdic/data_fmnist/fashion_base.bin. #points: 60000. #dim: 784.
# Wrote 6012 points to sample file: /home/cc/hpdic/data_fmnist/indices/diskann_base_R32_L50_B2G_sample_data.bin
# Indexing time: 1336.21
# (venv) cc@uc-ssd:~/hpdic/AdaDisk/experiments/fmnist$     