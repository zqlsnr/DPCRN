#include <string>
using namespace std;

const float win_sin_512[512] = {
0.0030679568,0.009203754,0.015339206,0.02147408,0.027608145,0.033741172,0.039872926,0.04600318,0.052131705,0.058258265,
0.06438263,0.070504576,0.076623864,0.08274026,0.08885355,0.0949635,0.10106986,0.10717242,0.11327095,0.119365215,
0.12545498,0.13154003,0.13762012,0.14369503,0.14976454,0.1558284,0.1618864,0.16793829,0.17398387,0.1800229,
0.18605515,0.1920804,0.1980984,0.20410897,0.21011184,0.2161068,0.22209363,0.22807208,0.23404196,0.24000302,
0.24595505,0.2518978,0.2578311,0.26375467,0.2696683,0.27557182,0.28146493,0.28734747,0.29321915,0.29907984,
0.30492923,0.31076714,0.31659338,0.3224077,0.32820985,0.33399966,0.33977687,0.34554133,0.35129276,0.35703096,
0.36275572,0.36846682,0.37416407,0.3798472,0.38551605,0.39117038,0.39681,0.40243465,0.40804416,0.41363832,
0.4192169,0.42477968,0.4303265,0.4358571,0.44137126,0.44686884,0.45234957,0.4578133,0.4632598,0.46868882,
0.4741002,0.47949377,0.48486924,0.49022648,0.49556527,0.50088537,0.50618666,0.5114688,0.5167318,0.5219753,
0.52719915,0.5324031,0.53758705,0.5427508,0.54789406,0.5530167,0.5581185,0.56319934,0.56825894,0.57329714,
0.57831377,0.58330864,0.5882816,0.5932323,0.5981607,0.6030666,0.6079498,0.6128101,0.6176473,0.62246126,
0.6272518,0.63201874,0.63676184,0.64148104,0.64617604,0.65084666,0.65549284,0.66011435,0.664711,0.6692826,
0.673829,0.67835003,0.68284553,0.68731534,0.6917592,0.6961771,0.7005688,0.70493406,0.7092728,0.71358484,
0.71787006,0.7221282,0.7263591,0.73056275,0.7347389,0.7388873,0.74300796,0.7471006,0.75116515,0.7552014,
0.7592092,0.7631884,0.7671389,0.7710605,0.7749531,0.7788165,0.7826506,0.7864552,0.7902302,0.7939755,
0.79769087,0.80137616,0.80503136,0.80865616,0.8122506,0.81581444,0.8193475,0.8228498,0.82632107,0.8297612,
0.8331702,0.83654773,0.8398938,0.84320825,0.8464909,0.84974176,0.8529606,0.85614735,0.8593018,0.86242396,
0.8655136,0.8685707,0.8715951,0.87458664,0.8775453,0.8804709,0.88336337,0.88622254,0.88904834,0.8918407,
0.8945995,0.89732456,0.9000159,0.9026733,0.90529674,0.9078861,0.9104413,0.9129622,0.9154487,0.9179008,
0.9203183,0.9227011,0.92504925,0.9273625,0.9296409,0.9318843,0.9340925,0.93626565,0.93840355,0.94050604,
0.9425732,0.9446048,0.9466009,0.9485614,0.95048606,0.952375,0.9542281,0.95604527,0.95782644,0.95957154,
0.96128047,0.96295327,0.9645898,0.96619,0.9677538,0.96928126,0.97077215,0.9722265,0.97364426,0.97502536,
0.97636974,0.97767735,0.9789482,0.9801821,0.9813792,0.9825393,0.9836624,0.9847485,0.9857975,0.9868094,
0.98778415,0.98872167,0.989622,0.9904851,0.99131083,0.9920993,0.9928504,0.9935641,0.99424046,0.9948793,
0.9954808,0.9960447,0.9965711,0.99706006,0.99751145,0.9979253,0.99830157,0.99864024,0.9989413,0.99920475,
0.9994306,0.9996188,0.9997694,0.99988234,0.9999576,0.9999953,0.9999953,0.9999576,0.99988234,0.9997694,
0.9996188,0.9994306,0.99920475,0.9989413,0.99864024,0.99830157,0.9979253,0.99751145,0.99706006,0.9965711,
0.9960447,0.9954808,0.9948793,0.99424046,0.9935641,0.9928504,0.9920993,0.99131083,0.9904851,0.989622,
0.98872167,0.98778415,0.9868094,0.9857975,0.9847485,0.9836624,0.9825393,0.9813792,0.9801821,0.9789482,
0.97767735,0.97636974,0.97502536,0.97364426,0.9722265,0.97077215,0.96928126,0.9677538,0.96619,0.9645898,
0.96295327,0.96128047,0.95957154,0.95782644,0.95604527,0.9542281,0.952375,0.95048606,0.9485614,0.9466009,
0.9446048,0.9425732,0.94050604,0.93840355,0.93626565,0.9340925,0.9318843,0.9296409,0.9273625,0.92504925,
0.9227011,0.9203183,0.9179008,0.9154487,0.9129622,0.9104413,0.9078861,0.90529674,0.9026733,0.9000159,
0.89732456,0.8945995,0.8918407,0.88904834,0.88622254,0.88336337,0.8804709,0.8775453,0.87458664,0.8715951,
0.8685707,0.8655136,0.86242396,0.8593018,0.85614735,0.8529606,0.84974176,0.8464909,0.84320825,0.8398938,
0.83654773,0.8331702,0.8297612,0.82632107,0.8228498,0.8193475,0.81581444,0.8122506,0.80865616,0.80503136,
0.80137616,0.79769087,0.7939755,0.7902302,0.7864552,0.7826506,0.7788165,0.7749531,0.7710605,0.7671389,
0.7631884,0.7592092,0.7552014,0.75116515,0.7471006,0.74300796,0.7388873,0.7347389,0.73056275,0.7263591,
0.7221282,0.71787006,0.71358484,0.7092728,0.70493406,0.7005688,0.6961771,0.6917592,0.68731534,0.68284553,
0.67835003,0.673829,0.6692826,0.664711,0.66011435,0.65549284,0.65084666,0.64617604,0.64148104,0.63676184,
0.63201874,0.6272518,0.62246126,0.6176473,0.6128101,0.6079498,0.6030666,0.5981607,0.5932323,0.5882816,
0.58330864,0.57831377,0.57329714,0.56825894,0.56319934,0.5581185,0.5530167,0.54789406,0.5427508,0.53758705,
0.5324031,0.52719915,0.5219753,0.5167318,0.5114688,0.50618666,0.50088537,0.49556527,0.49022648,0.48486924,
0.47949377,0.4741002,0.46868882,0.4632598,0.4578133,0.45234957,0.44686884,0.44137126,0.4358571,0.4303265,
0.42477968,0.4192169,0.41363832,0.40804416,0.40243465,0.39681,0.39117038,0.38551605,0.3798472,0.37416407,
0.36846682,0.36275572,0.35703096,0.35129276,0.34554133,0.33977687,0.33399966,0.32820985,0.3224077,0.31659338,
0.31076714,0.30492923,0.29907984,0.29321915,0.28734747,0.28146493,0.27557182,0.2696683,0.26375467,0.2578311,
0.2518978,0.24595505,0.24000302,0.23404196,0.22807208,0.22209363,0.2161068,0.21011184,0.20410897,0.1980984,
0.1920804,0.18605515,0.1800229,0.17398387,0.16793829,0.1618864,0.1558284,0.14976454,0.14369503,0.13762012,
0.13154003,0.12545498,0.119365215,0.11327095,0.10717242,0.10106986,0.0949635,0.08885355,0.08274026,0.076623864,
0.070504576,0.06438263,0.058258265,0.052131705,0.04600318,0.039872926,0.033741172,0.027608145,0.02147408,0.015339206,
0.009203754,0.0030679568
};