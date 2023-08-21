from typing import Dict

import SimpleITK as sitk
import torch
import os
import numpy as np

from base_algorithm import BaseSynthradAlgorithm


class SynthradAlgorithm(BaseSynthradAlgorithm):
    """
    This class implements a simple synthetic CT generation algorithm that segments all values greater than 2 in the input image.

    Author: Suraj Pai (b.pai@maastrichtuniversity.nl)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = os.path.join(os.path.dirname(__file__), 'synthrad_model.pt')

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, input_dict: Dict[str, sitk.Image]) -> sitk.Image:
        """
        Generates a synthetic CT image from the given input image and mask.

        Parameters
        ----------
        input_dict : Dict[str, SimpleITK.Image]
            A dictionary containing two keys: "image" and "mask". The value for each key is a SimpleITK.Image object representing the input image and mask respectively.

        Returns
        -------
        SimpleITK.Image
            The generated synthetic CT image.

        Raises
        ------
        AssertionError:
            If the keys of `input_dict` are not ["image", "mask"]
        """
        assert list(input_dict.keys()) == ["image", "mask", "region"]

        # You may use the region information to generate the synthetic CT image if needed 
        region = input_dict["region"]
        print("Scan region: ", region)
        mr_sitk = input_dict["image"]
        mask_sitk = input_dict["mask"]

        # convert sitk images to np arrays
        mask_np = sitk.GetArrayFromImage(mask_sitk).astype("float32")
        mr_np = sitk.GetArrayFromImage(mr_sitk).astype("float32")


        if 'head' in region.lower():
            region_cond_np = np.ones_like(x_input)
        else:
            region_cond_np = -1 * np.ones_like(x_input)
        region_cond_np *= mask_np

        print("Using device: ", self.device)

        x_input = torch.Tensor(np.stack([mr_np, mask_np, region_cond_np], axis=0), device=self.device)

        output = self.model.predict(x_input)

        # convert tensor back to np array
        sCT = output.cpu().numpy()

        sCT_sitk = sitk.GetImageFromArray(sCT)
        sCT_sitk.CopyInformation(mr_sitk)

        return sCT_sitk


if __name__ == "__main__":
    # Run the algorithm on the default input and output paths specified in BaseSynthradAlgorithm.
    SynthradAlgorithm().process()
