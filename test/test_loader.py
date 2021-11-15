import os
import pathlib

import hannah_tvm.load as load


root_dir = path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def test_tflite():
    import tflite

    model_path = (
        pathlib.Path(root_dir)
        / "models"
        / "tiny_ml_perf"
        / "keyword_spotting"
        / "kws_ref_model.tflite"
    )
    assert model_path.exists()

    with model_path.open("rb") as model_file:
        modelBuf = model_file.read()
        tflModel = tflite.Model.GetRootAsModel(modelBuf, 0)

        print(tflModel)

        assert tflModel.SubgraphsLength() == 1
        g = tflModel.Subgraphs(0)

        print(g)


if __name__ == "__main__":
    test_tflite()
