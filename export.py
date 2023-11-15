import paddle

paddle.set_grad_enabled(mode=False)

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
            dtype=tenInput.dtype
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


if __name__ == "__main__":
    try:
        model = Network()
        tenInput = paddle.randn([1, 3, 320, 480])
        input_spec = list(paddle.static.InputSpec.from_tensor(paddle.to_tensor(t)) for t in (tenInput, ))
        paddle.jit.save(model, input_spec=input_spec, path="./model")
        print('[JIT] paddle.jit.save successed.')
        exit(0)
    except Exception as e:
        print('[JIT] paddle.jit.save failed.')
        raise e
