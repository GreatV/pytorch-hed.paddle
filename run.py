import paddle
import getopt
import numpy
import PIL
import PIL.Image
import sys

paddle.set_grad_enabled(mode=False)

arguments_strModel = "bsds500"
arguments_strIn = "./images/sample.png"
arguments_strOut = "./out.png"
for strOption, strArgument in getopt.getopt(
    sys.argv[1:], "", [(strParameter[2:] + "=") for strParameter in sys.argv[1::2]]
)[0]:
    if strOption == "--model" and strArgument != "":
        arguments_strModel = strArgument
    if strOption == "--in" and strArgument != "":
        arguments_strIn = strArgument
    if strOption == "--out" and strArgument != "":
        arguments_strOut = strArgument

class Network(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.netVggOne = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
        )
        self.netVggTwo = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
        )
        self.netVggThr = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
        )
        self.netVggFou = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
        )
        self.netVggFiv = paddle.nn.Sequential(
            paddle.nn.MaxPool2D(kernel_size=2, stride=2),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            paddle.nn.ReLU(),
        )
        self.netScoreOne = paddle.nn.Conv2D(
            in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreTwo = paddle.nn.Conv2D(
            in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreThr = paddle.nn.Conv2D(
            in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFou = paddle.nn.Conv2D(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netScoreFiv = paddle.nn.Conv2D(
            in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0
        )
        self.netCombine = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0
            ),
            paddle.nn.Sigmoid(),
        )

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - paddle.to_tensor(
            data=[104.00698793, 116.66876762, 122.67891434],
            dtype=tenInput.dtype,
            place=tenInput.place,
        ).reshape([1, 3, 1, 1])
        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)
        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)
        tenScoreOne = paddle.nn.functional.interpolate(
            x=tenScoreOne,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreTwo = paddle.nn.functional.interpolate(
            x=tenScoreTwo,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreThr = paddle.nn.functional.interpolate(
            x=tenScoreThr,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFou = paddle.nn.functional.interpolate(
            x=tenScoreFou,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        tenScoreFiv = paddle.nn.functional.interpolate(
            x=tenScoreFiv,
            size=(tenInput.shape[2], tenInput.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return self.netCombine(
            paddle.concat(
                x=[tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv],
                axis=1,
            )
        )


netNetwork = None


def estimate(tenInput):
    global netNetwork
    if netNetwork is None:
        netNetwork = Network().eval()
    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]
    assert intWidth == 480
    assert intHeight == 320
    return netNetwork(tenInput.view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


if __name__ == "__main__":
    tenInput = paddle.to_tensor(
        data=numpy.ascontiguousarray(
            numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(numpy.float32)
            * (1.0 / 255.0)
        ),
        dtype="float32",
    )
    tenOutput = estimate(tenInput)
    PIL.Image.fromarray(
        (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(
            numpy.uint8
        )
    ).save(arguments_strOut)
