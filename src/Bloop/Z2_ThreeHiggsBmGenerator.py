"""Generate benchmarks at 0T satisfying some constraints.
If the functions return True then the benchmark point is allowed by those constraints
Variables that are bools are denoted with a b prefix"""

import math as m
import numpy as np

import json
from pathlib import Path
from os.path import join
from glob import glob

from Bloop.ParsedExpression import ParsedExpression
from Bloop.EffectivePotential import cNlopt
from Bloop.PDGData import mHiggs, higgsVEV


def bIsBounded(params):
    ## Taking equations 26-31 from the draft that ensure the potential is bounded from below.
    if not params["lamda11"] > 0:
        return False
    if not params["lamda22"] > 0:
        return False
    if not params["lamda33"] > 0:
        return False

    lamdax = params["lamda12"] + min(
        0, params["lamda12p"] - 2 * m.sqrt(params["lamda1Re"] ** 2 + params["lamda1Im"] ** 2)
    )
    lamday = params["lamda31"] + min(
        0, params["lamda31p"] - 2 * m.sqrt(params["lamda3Re"] ** 2 + params["lamda3Im"] ** 2)
    )
    lamdaz = params["lamda23"] + min(
        0, params["lamda23p"] - 2 * m.sqrt(params["lamda2Re"] ** 2 + params["lamda2Im"] ** 2)
    )
    if not lamdax > -2 * m.sqrt(params["lamda11"] * params["lamda22"]):
        return False
    if not lamday > -2 * m.sqrt(params["lamda11"] * params["lamda33"]):
        return False
    if not lamdaz > -2 * m.sqrt(params["lamda22"] * params["lamda33"]):
        return False
    if not (
        m.sqrt(params["lamda33"]) * lamdax
        + m.sqrt(params["lamda11"]) * lamdaz
        + m.sqrt(params["lamda22"]) * lamday
        >= 0
        or params["lamda33"] * lamdax**2
        + params["lamda11"] * lamdaz**2
        + params["lamda22"] * lamday**2
        - params["lamda11"] * params["lamda22"] * params["lamda33"]
        - 2 * lamdax * lamday * lamdaz
        < 0
    ):
        return False
    return True


def bPhysicalMinimum(nloptInst, potential, params):
    ## Move these to user args or somewhere out of the way
    minimumInitialGuesses = [
        [0, 0, 0],
        [0, 0, 246],
        [100, 100, 100],
        [-100, 100, 100],
        [50, 50, 50],
        [299, 299, 299],
        [-299, 299, 299],
    ]

    ## I don't like having the function wrapped here,
    ## should be defined once at a higher level, not every func call but params
    ## makes that non-trivial
    def potentialWrapped(fields, _):
        return potential(fields, params)

    minValue = nloptInst.nloptGlobal(potentialWrapped, minimumInitialGuesses[0])[1]

    for guess in minimumInitialGuesses:
        minLocationTemp, minValueTemp = nloptInst.nloptLocal(potentialWrapped, guess)

        if minValueTemp < minValue:
            minLocation, minValue = minLocationTemp, minValueTemp

    return np.all(np.isclose(minLocation, [0, 0, higgsVEV], atol=1))


def _lagranianParamGen(
    mS1, delta12, delta1c, deltac, ghDM, thetaCPV, darkHieracy, bmNumber
):
    ## SM params
    vsq = higgsVEV**2
    mu3sq = mHiggs**2 / 2
    lamda33 = mu3sq / vsq

    ## ~~~ USER BSM PHYSICS ~~~
    mS2 = delta12 + mS1
    mSpm1 = delta1c + mS1
    mSpm2 = deltac + mSpm1
    mu12sq = (mSpm2**2 - mSpm1**2) / 2

    sinTheta, cosTheta = m.sin(thetaCPV), m.cos(thetaCPV)
    lamda2absInsideSqR = (
        (2.0 * mu12sq * cosTheta) ** 2
        + (mS2**2 - mS1**2) ** 2
        - (mSpm2**2 - mSpm1**2) ** 2
    )
    if lamda2absInsideSqR < 0:
        return False

    lamda2Abs = (mu12sq * cosTheta + m.sqrt(lamda2absInsideSqR) / 4) / vsq
    lamda2Re = lamda2Abs * cosTheta
    lamda2Im = lamda2Abs * sinTheta

    alpha = (
        -mu12sq
        + vsq * lamda2Abs * cosTheta
        - m.sqrt(
            mu12sq**2 + vsq**2 * lamda2Abs**2 - 2.0 * vsq * mu12sq * lamda2Abs * cosTheta
        )
    ) / ((vsq * lamda2Abs * sinTheta) + 1e-100)
    mu2sq = (
        vsq / 2.0 * ghDM
        - vsq
        / (alpha**2 + 1.0)
        * (2.0 * alpha * sinTheta + (alpha**2 - 1.0) * cosTheta)
        * lamda2Abs
        - (mS2**2 + mS1**2) / 2
    )
    lamda23 = (2.0 * mu2sq + mSpm2**2 + mSpm1**2) / vsq
    lamda23p = (mS2**2 + mS1**2 - mSpm2**2 - mSpm1**2) / vsq

    paramsDict = {
        "bmNumber": bmNumber,
        "RGScale": 91.1876,
        "bmInput": {
            "thetaCPV": thetaCPV,
            "ghDM": ghDM,
            "mS1": mS1,
            "delta12": delta12,
            "delta1c": delta1c,
            "deltac": deltac,
            "darkHieracy": darkHieracy,
        },
        "massTerms": {
            "mu12sqRe": mu12sq,
            "mu12sqIm": 0,
            "mu2sq": mu2sq,
            "mu3sq": mu3sq,
            "mu1sq": darkHieracy * mu2sq,
        },
        "couplingValues": {
            "lamda1Re": 0.1,
            "lamda1Im": 0,
            "lamda2Re": lamda2Abs * cosTheta,
            "lamda2Im": lamda2Abs * sinTheta,
            "lamda11": 0.11,
            "lamda22": 0.12,
            "lamda12": 0.13,
            "lamda12p": 0.14,
            "lamda23": lamda23,
            "lamda23p": lamda23p,
            "lamda3Re": darkHieracy * lamda2Re,
            "lamda3Im": darkHieracy * lamda2Im,
            "lamda31": darkHieracy * lamda23,
            "lamda31p": darkHieracy * lamda23p,
            "lamda33": lamda33,
        },
    }
    return paramsDict


def checkPhysical(params, nloptInst, potential, chargedMassMatrix, neutralMassMatrix):
    params["v1"] = 0
    params["v2"] = 0
    params["v3"] = 246.22
    if not bIsBounded(params):
        return False

    chargedEigenValues = np.linalg.eigvalsh(chargedMassMatrix.evaluate(params))
    ## Enforces positive charged masses (tolerance to handle goldstone bosons)
    if not np.all(chargedEigenValues >= -1e-20):
        return False
    ## ASSUMING ONLY TWO GOLDSTONES check if masses are heavy enough to avoid dection
    ## (above 90GeV (8100 = 90**2 - avoids sqrt))
    if not np.all(chargedEigenValues[2:] >= 8100):
        return False

    neutralEigenValues = np.linalg.eigvalsh(neutralMassMatrix.evaluate(params))
    ## Enforces positive neutral masses (tolerance to handle goldstone bosons)
    if not np.all(neutralEigenValues >= -1e-20):
        return False
    ## ASSUMING ONLY ONE GOLDSTONE check if masses are heavy enough to avoid
    ## Higgs decaying to light particle
    ## (above 63GeV (3969 = 63**2 - avoids sqrt))
    if not np.all(neutralEigenValues[1:] >= 3969):
        return False

    if not bPhysicalMinimum(nloptInst, potential, params):
        return False

    return True


def _randomBmParam(
    randomNum, nloptInst, potential, chargedMassMatrix, neutralMassMatrix
):
    bmdictList = []
    ## TODO put in some upper limit for this while loop
    darkHieracy = 1
    ## Run in parallel?
    while len(bmdictList) < randomNum:
        mS1 = np.random.uniform(63, 100)
        delta12 = np.random.uniform(5, 100)
        delta1c = np.random.uniform(5, 100)
        deltac = np.random.uniform(5, 100)
        ghDM = np.random.uniform(0, 1)
        thetaCPV = np.random.uniform(np.pi / 2, 3 * np.pi / 2)

        bmParams = _lagranianParamGen(
            mS1, delta12, delta1c, deltac, ghDM, thetaCPV, darkHieracy, len(bmdictList)
        )

        if bmParams:
            if checkPhysical(
                bmParams["massTerms"] | bmParams["couplingValues"],
                nloptInst,
                potential,
                chargedMassMatrix,
                neutralMassMatrix,
            ):
                bmdictList.append(bmParams)

    return bmdictList


def _handPickedBm(nloptInst, potential, chargedMassMatrix, neutralMassMatrix):
    bmdictList = []
    ## Move this to userArg or something
    bmInputList = [
        [300, 0, 0, 0, 0.0, 0.0, 1],
        [67, 4.0, 50.0, 1.0, 0.0, 2.0 * np.pi / 3, 1],
        [70, 12.0, 50.0, 1.0, 0.0, 2.0 * np.pi / 3, 1],
        [75, 55.0, 50.0, 1.0, 0.0, 2.0 * np.pi / 3, 1],
        [74, 55.0, 50.0, 15.0, 0.0, 2.0 * np.pi / 3, 1],
        [90, 5.0, 1.0, 1.0, 0.0, 2.0 * np.pi / 3, 1],
        [90, 55.0, 1.0, 1.0, 0.0, 2.0 * np.pi / 3, 1],
        [90, 55.0, 1.0, 22.0, 0.0, 2.0 * np.pi / 3, 1],
    ]

    for bmInput in bmInputList:
        bmParams = _lagranianParamGen(*bmInput, len(bmdictList))
        if bmParams:
            if checkPhysical(
                bmParams["massTerms"] | bmParams["couplingValues"],
                nloptInst,
                potential,
                chargedMassMatrix,
                neutralMassMatrix,
            ):
                bmdictList.append(bmParams)

    return bmdictList


def _strongSubSet(prevResultDir):
    bmdictList = []

    for fileName in glob(join(prevResultDir, "*.json")):
        with open(fileName, "r") as fp:
            resultDic = json.load(fp)

        if resultDic["strong"] > 0.6:
            bmdictList.append(
                _lagranianParamGen(
                    resultDic["bmInput"]["mS1"],
                    resultDic["bmInput"]["delta12"],
                    resultDic["bmInput"]["delta1c"],
                    resultDic["bmInput"]["deltac"],
                    resultDic["bmInput"]["ghDM"],
                    resultDic["bmInput"]["thetaCPV"],
                    resultDic["bmInput"]["darkHieracy"],
                    resultDic["bmNumber"],
                )
            )

    return bmdictList


def generateBenchmarks(args):
    (output_file := Path(args.benchmarkFile)).parent.mkdir(exist_ok=True, parents=True)

    ## Need to take these from args
    nloptInst = cNlopt(
        config={
            "nbrVars": 3,
            "absGlobalTol": 0.5,
            "relGlobalTol": 0.5,
            "absLocalTol": 0.5,
            "relLocalTol": 0.5,
            "varLowerBounds": [-300, 0, 0],
            "varUpperBounds": [300, 300, 300],
        }
    )
    with open(args.pythonisedExpressionsFile, "r") as fp:
        parsedExpressions = json.load(fp)
    ## Take the pythonised tree level potential we've generated
    treeLevel = ParsedExpression(parsedExpressions["veff"]["expressions"][0], None)
    chargedMassMatrix = ParsedExpression(
        parsedExpressions["scalarMassMatrices"]["expressions"][0], None
    )
    neutralMassMatrix = ParsedExpression(
        parsedExpressions["scalarMassMatrices"]["expressions"][1], None
    )

    ## Feels weird to have nested function but not sure how else to go until can
    ## use arrays with nlopt
    def potential(fields, params):
        params["v1"] = fields[0]
        params["v2"] = fields[1]
        params["v3"] = fields[2]
        return treeLevel.evaluate(params)

    if args.benchmarkType == "randomSSS":
        with open(output_file, "w") as fp:
            json.dump(_strongSubSet(args.prevResultDir), fp, indent=4)
        return

    elif args.benchmarkType == "handPicked":
        with open(output_file, "w") as fp:
            json.dump(
                _handPickedBm(nloptInst, potential, chargedMassMatrix, neutralMassMatrix),
                fp,
                indent=4,
            )

    elif args.benchmarkType == "random":
        with open(output_file, "w") as fp: 
            json.dump(
                _randomBmParam(
                    args.randomNum,
                    nloptInst,
                    potential,
                    chargedMassMatrix,
                    neutralMassMatrix,
                ),
                fp,
                indent=4,
        )

    return


from unittest import TestCase


class BmGeneratorUnitTests(TestCase):
    def test_bIsBoundedFalse(self):
        source = {
            "mu12sqRe": 12724.595168103033,
            "mu12sqIm": 0,
            "mu2sq": -20604.175986862854,
            "mu3sq": 7812.5,
            "mu1sq": -20604.175986862854,
            "lamda1Re": 0.1,
            "lamda1Im": 0.0,
            "lamda2Re": 0.08368703688875163,
            "lamda2Im": 0.05054893896685599,
            "lamda11": 0.11,
            "lamda22": 0.12,
            "lamda12": 0.13,
            "lamda12p": 0.14,
            "lamda23": 0.13184965469208287,
            "lamda23p": -0.10179114069822925,
            "lamda3Re": 0.08368703688875163,
            "lamda3Im": 0.05054893896685599,
            "lamda31": 0.13184965469208287,
            "lamda31p": -0.10179114069822925,
            "lamda33": 0.12886749199352251,
        }
        self.assertEqual(False, bIsBounded(source))

    def test_bIsBoundedTrue(self):
        source = {
            "mu12sqRe": 4799.941333804141,
            "mu12sqIm": 0,
            "mu2sq": -11505.594825493996,
            "mu3sq": 7812.5,
            "mu1sq": -11505.594825493996,
            "lamda1Re": 0.1,
            "lamda1Im": 0.0,
            "lamda2Re": 0.010823997158779158,
            "lamda2Im": 0.017584425457946057,
            "lamda11": 0.11,
            "lamda22": 0.12,
            "lamda12": 0.13,
            "lamda12p": 0.14,
            "lamda23": 0.33218995023667736,
            "lamda23p": -0.29472720912675215,
            "lamda3Re": 0.010823997158779158,
            "lamda3Im": 0.017584425457946057,
            "lamda31": 0.33218995023667736,
            "lamda31p": -0.29472720912675215,
            "lamda33": 0.12886749199352251,
        }
        self.assertEqual(True, bIsBounded(source))

    def test_lagranianParamGen(self):
        reference = {
            "bmNumber": 0,
            "RGScale": 91.1876,
            "bmInput": {
                "thetaCPV": 3.11308902835221,
                "ghDM": 0.15520161865427817,
                "mS1": 89.15641588128479,
                "delta12": 87.17952518246265,
                "delta1c": 14.020273320699415,
                "deltac": 5.129099092707543,
                "darkHieracy": 1,
            },
            "massTerms": {
                "mu12sqRe": 542.3572917258725,
                "mu12sqIm": 0,
                "mu2sq": -9572.921254799061,
                "mu3sq": 7837.46120740695,
                "mu1sq": -9572.921254799061,
            },
            "couplingValues": {
                "lamda1Re": 0.1,
                "lamda1Im": 0,
                "lamda2Re": -0.08646892283299933,
                "lamda2Im": 0.0024653454694036642,
                "lamda11": 0.11,
                "lamda22": 0.12,
                "lamda12": 0.13,
                "lamda12p": 0.14,
                "lamda23": 0.05327468098550607,
                "lamda23p": 0.2749344421058253,
                "lamda3Re": -0.08646892283299933,
                "lamda3Im": 0.0024653454694036642,
                "lamda31": 0.05327468098550607,
                "lamda31p": 0.2749344421058253,
                "lamda33": 0.12927959478844356,
            },
        }
        source = (
            89.15641588128479,
            87.17952518246265,
            14.020273320699415,
            5.129099092707543,
            0.15520161865427817,
            3.11308902835221,
            1,
            0,
        )
        self.assertEqual(reference, _lagranianParamGen(*source))
