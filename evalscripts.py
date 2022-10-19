evalscript_raw = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04",
                        "B03",
                        "B08",
                        "dataMask",
                        "CLM"],
                units: "DN"
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1)  {
            if (sample.CLM == 0) {
                let NDVI = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)
                let GNDVI = (sample.B08 - sample.B03) / (sample.B08 + sample.B03)
                return [NDVI, GNDVI]
            } else {
                return [NaN, NaN]
            }
        } else {
            return [NaN, NaN]
        }
    }
"""


evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B07"]
            }],
            output: {
                bands: 1
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B07, sample.B07, sample.B07];
    }
"""