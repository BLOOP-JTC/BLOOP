import json
from sympy.parsing.mathematica import parse_mathematica
from numpy import euler_gamma, pi
from pathlib import Path
from importlib.resources import files
import unicodedata
import re

from Veff_generation import generate_veff_module, compile_veff_submodule

def getLines(relativePathToResource):
    with open(files(__package__) / relativePathToResource, "r", encoding="utf-8") as fp:
        return fp.readlines()

def getLinesJSON(relativePathToResource):
    with open(files(__package__) / relativePathToResource, "r") as fp:
        return json.load(fp)


def replaceGreekSymbols(string):
    def replaceGreekCharacter(match):
        characterData = unicodedata.name(match.group(0)).split()
        
        if 'SMALL' in characterData:
            return characterData[-1].lower()
        
        elif 'CAPITAL' in characterData:
            return characterData[-1].capitalize()
    
    greekCharacters = r'[\u0391-\u03A9\u03B1-\u03C9]'
    result = re.sub(greekCharacters, replaceGreekCharacter, string)
    
    return result


def replaceSymbolsConst(string):
    return (
        string.replace("Pi", str(pi))
        .replace("EulerGamma", str(euler_gamma))
        .replace("Glaisher", "1.28242712910062")
    )


def removeSuffices(string):
    return string.replace("^2", "sq")


def replaceSymbolsWithIndices(expression, symbols):
    expression = replaceGreekSymbols(expression)
    ## Reverse needed to deal with lam23 and lam23p i.e. substring replaces larger full string
    for idx, symbol in enumerate(sorted(symbols, reverse=True)):
        expression = expression.replace(symbol, f"params[{idx}]")

    return expression


def pythoniseExpressionArray(line, allSymbols):
    identifier, line = (
        map(str.strip, line.split("->")) if ("->" in line) else ("missing", line)
    )

    identifier = removeSuffices(replaceGreekSymbols(identifier))
    expression = parse_mathematica(replaceSymbolsConst(replaceGreekSymbols(line)))
    symbols = [str(symbol) for symbol in expression.free_symbols]

    return {
        "identifier": identifier,
        "expression": replaceSymbolsWithIndices(str(expression), allSymbols),
        "symbols": sorted(symbols),
    }


def pythoniseExpression(line):
    identifier, line = (
        map(str.strip, line.split("->")) if ("->" in line) else ("missing", line)
    )

    identifier = removeSuffices(replaceGreekSymbols(identifier))
    expression = parse_mathematica(replaceSymbolsConst(replaceGreekSymbols(line)))
    symbols = [str(symbol) for symbol in expression.free_symbols]

    return {
        "identifier": identifier,
        "expression": str(expression),
        "symbols": sorted(symbols),
    }


def pythoniseExpressionSystemArray(lines, allSymbols):
    return [pythoniseExpressionArray(line, allSymbols) for line in lines]


def pythoniseExpressionSystem(lines):
    return [pythoniseExpression(line) for line in lines]


def pythoniseMatrix(lines):
    return [
        [symbol.strip() for symbol in line.strip().strip("}").strip("{").split(",")]
        for line in lines
    ]


def pythoniseMathematica(args):
    veffLines = getLines(args.loFile)
    veffLines += getLines(args.nloFile)
    if args.loopOrder >= 2:
        veffLines += getLines(args.nnloFile)
    
    scalarRotationMatrix = getLinesJSON(args.scalarRotationMatrixFile)
    allSymbols = getLinesJSON(args.allSymbolsFile) + ["missing"]
    allSymbols = sorted(
        [replaceGreekSymbols(symbol) for symbol in allSymbols], reverse=True
    )

    ## Move get lines to the functions? -- Would need to rework veffLines in this case
    ## Not ideal to have nested dicts but is future proof for when we move to arrays
    expressionDict = {
        "bounded": {
            "expressions": pythoniseExpressionSystemArray(
                getLines(args.boundedConditions), allSymbols
            ),
            "fileName": "bounded",
        },
        "betaFunctions4D": {
            "expressions": pythoniseExpressionSystemArray(
                getLines(args.betaFunctions4DFile), allSymbols
            ),
            "fileName": args.betaFunctions4DFile,
        },
        "hardToSoft": {
            "expressions": pythoniseExpressionSystemArray(
                getLines(args.hardToSoftFile), allSymbols
            ),
            "fileName": args.hardToSoftFile,
        },
        "softScaleRGE": {
            "expressions": pythoniseExpressionSystemArray(
                getLines(args.softScaleRGEFile), allSymbols
            ),
            "fileName": args.softScaleRGEFile,
        },
        "softToUltraSoft": {
            "expressions": pythoniseExpressionSystemArray(
                getLines(args.softToUltraSoftFile), allSymbols
            ),
            "fileName": args.softToUltraSoftFile,
        },
        "vectorMassesSquared": {
            "expressions": pythoniseExpressionSystem(
                getLines(args.vectorMassesSquaredFile)
            ),
            "fileName": args.vectorMassesSquaredFile,
        },
        "vectorShorthands": {
            "expressions": pythoniseExpressionSystem(
                getLines(args.vectorShortHandsFile)
            ),
            "fileName": args.vectorShortHandsFile,
        },
        "veff": {
            "expressions": pythoniseExpressionSystem(veffLines),
            "fileName": "Combined Veff files",
        },
        
        "scalarMassMatrices": {
            "expressions": pythoniseExpressionSystem(getLines(args.scalarMassMatrixFile)),
            "fileName": args.scalarMassMatrixFile
        },
        
        
        
        "scalarRotationMatrix": {
            "scalarRotationMatrix": scalarRotationMatrix,
            "fileName": args.scalarRotationMatrixFile,
        },
        "allSymbols": {
            "allSymbols": allSymbols,
            "fileName": args.allSymbolsFile,
        },
        "lagranianVariables": {
            "lagranianVariables": getLinesJSON(args.lagranianVariablesFile),
            "fileName": args.lagranianVariablesFile 
        },
        "scalarMassNames": {
            "scalarMassNames": getLinesJSON(args.scalarMassNamesFile),
            "fileName": args.scalarMassNamesFile 
        },
    }

    expressionDict["scalarPermutationMatrix"] = (
        []
        if args.scalarPermutationMatrixFile.lower() == "none"
        else getLinesJSON(args.scalarPermutationMatrixFile)
    )
    
    if args.bCython:
        generate_veff_module(
            args, 
            allSymbols, 
            args.scalarMassMatrixFile, 
            expressionDict["scalarMassNames"]["scalarMassNames"],
            args.scalarPermutationMatrixFile, 
            args.scalarRotationMatrixFile, 
            expressionDict["vectorMassesSquared"]["expressions"],
            expressionDict["vectorShorthands"]["expressions"],
        )

        compile_veff_submodule(args)    
    
    else:
        expressionDict["veffArray"] = {
            "expressions": pythoniseExpressionSystemArray(veffLines, allSymbols),
            "fileName": "Combined Veff files",
        }

    (outputFile := Path(args.pythonisedExpressionsFile)).parent.mkdir(
        exist_ok=True, parents=True
    )   
    with open(outputFile, "w") as fp:
        json.dump(expressionDict, fp, indent=4)
    


from unittest import TestCase


class PythoniseMathematicaUnitTests(TestCase):
    def test_replaceGreekSymbols(self):
        reference = ["lam", "lam lam", "mu", "mu mu", "lam mu", "mu lam"]
        source = ["λ", "λ λ", "μ", "μ μ", "λ μ", "μ λ"]

        self.assertEqual(
            reference, [replaceGreekSymbols(sourceString) for sourceString in source]
        )

    def test_removeSuffices(self):
        reference = ["myVarsq", "sqmyVar"]

        source = ["myVar^2", "^2myVar"]

        self.assertEqual(
            reference, [removeSuffices(sourceString) for sourceString in source]
        )

    def test_pythoniseExpression(self):
        reference = {
            "expression": "0.07957747154594767*sqrt(lam) + log(mssq)",
            "identifier": "Identifier",
            "symbols": ["lam", "mssq"],
        }

        source = "Identifier -> Sqrt[λ] / (4 * Pi) + Log[mssq]"

        self.assertEqual(reference, pythoniseExpression(source))

    def test_paseExpressionSystem(self):
        reference = [
            {
                "expression": "0.07957747154594767*sqrt(lam) + log(mssq)",
                "identifier": "Identifier",
                "symbols": ["lam", "mssq"],
            },
            {
                "expression": "0.07957747154594767*sqrt(lam) + log(mssq)",
                "identifier": "Identifier",
                "symbols": ["lam", "mssq"],
            },
            {
                "expression": "0.07957747154594767*sqrt(lam) + log(mssq)",
                "identifier": "Identifier",
                "symbols": ["lam", "mssq"],
            },
        ]

        source = [
            "Identifier -> Sqrt[λ] / (4 * Pi) + Log[mssq]",
            "Identifier -> Sqrt[λ] / (4 * Pi) + Log[mssq]",
            "Identifier -> Sqrt[λ] / (4 * Pi) + Log[mssq]",
        ]

        self.assertEqual(reference, pythoniseExpressionSystem(source))

    def test_pythoniseMatrix(self):
        reference = [["1", "0"], ["0", "0"]]
        source = ["{1, 0}", "{0, 0}"]

        self.assertEqual(reference, pythoniseMatrix(source))

