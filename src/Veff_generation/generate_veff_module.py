import os
from textwrap import dedent
from jinja2 import Environment
import numpy as np

import Bloop.PythoniseMathematica as PythoniseMathematica

def generate_veff_module(
    args, 
    allSymbols, 
    scalarMassMatrixFile,
    scalarMassNames,
):
    
    parent_dir = os.path.dirname(os.getcwd())
    data_dir   = os.path.join(parent_dir, 'src', 'Bloop')
    module_dir = os.path.join(parent_dir, 'src', 'Bloop', 'Veff')
    
    if not os.path.exists(module_dir):
        os.mkdir(module_dir)
    if args.verbose:
        print("Generating Veff submodule")
    
    loopOrder = args.loopOrder 
    
    veffFPs   = [args.loFile, args.nloFile]
    veffNames = ["lo", "nlo"]
    
    if loopOrder >1:
        veffFPs.append(args.nnloFile)
        veffNames.append("nnlo")
        
    for idx, name in enumerate(veffNames):
        generateVeffSubModule(
            name, 
            os.path.join(module_dir, f"{name}.pyx"), 
            os.path.join(data_dir, veffFPs[idx]), 
            allSymbols
        )
    
    generateDiagonalizeSubModule(
        os.path.join(module_dir, "eigen.pyx"), 
        allSymbols,
        os.path.join(data_dir, scalarMassMatrixFile),
        scalarMassNames,
    )

    generateVeffModule(
        os.path.join(module_dir, 'veff.py'), 
        loopOrder, 
        allSymbols
    )
    
    #================================ init file ==============================#
    with open(os.path.join(module_dir, '__init__.py'), 'w') as file:
        file.write("from .veff import *")
    
    #=============================== setup file ==============================#
    with open(os.path.join(module_dir, 'setup.py'), 'w') as file:
        file.writelines(Environment().from_string(dedent("""\
            #!/usr/bin/env python3
            # -*- coding: utf-8 -*-
            from setuptools import setup, Extension
            from Cython.Build import cythonize
            
            extensions = [Extension("lo", ["lo.pyx"])]
            {% if args.loopOrder >= 1 %}
            extensions.append(Extension("nlo", ["nlo.pyx"]))
            {% endif %}
            {% if args.loopOrder >= 2 %}
            extensions.append(Extension("nnlo", ["nnlo.pyx"]))
            {% endif %}
            extensions.append(Extension("eigen", ["eigen.pyx"]))

            setup(
                name="Veff_cython",
                ext_modules=cythonize(
                    extensions, compiler_directives={"language_level": "3"}
                ),
            )
            """
        )).render(args = args))
        
def generateVeffModule(filename, loopOrder, allSymbols):
    """Write a  function that imports veff submodules based on loopOrder,
    returns the evaluated submodules as a tuple.
    """
    with open(filename, 'w') as file:
        file.write(Environment().from_string(dedent(
        """\
        from .lo import lo
        from .nlo import nlo
        {%- if loopOrder > 1 %}
        from .nnlo import nnlo
        {%- endif %}
        from .eigen import eigen

        def eigen(*args):
            return eigen(args)
        
        def Veff(
        {%- for symbol in allSymbols %}
            {{ symbol }} = 1,
        {%- endfor %}
            ):
            val_lo = lo(
        {%- for symbol in allSymbols %}
                {{ symbol }},
        {%- endfor %}
            )
            
            val_nlo = nlo(
        {%- for symbol in allSymbols %}
                {{ symbol }},
        {%- endfor %}
            )
            
        {%- if loopOrder > 1 %}
            val_nnlo = nnlo(
        {%- for symbol in allSymbols %}
                {{ symbol }},
        {%- endfor %}
            )
            return (val_lo, val_nlo, val_nnlo)
        
        {%- else %}
            return (val_lo, val_nlo)
        {%- endif %}
        """)).render(loopOrder=loopOrder, allSymbols=allSymbols))
     

def generateVeffSubModule(name, moduleName, veffFp, allSymbols):
    # Creates a cython module with that computes an order of Veff
    with open(moduleName, 'w') as file:
    
        file.write(Environment().from_string(dedent("""\
            #cython: cdivision=False
            from libc.complex cimport csqrt
            from libc.complex cimport clog
            
            cpdef double complex {{ name }}(
            {%- for symbol in allSymbols %}
                double complex {{ symbol }},
            {%- endfor %}
                ):
                ## Calling _name decreases compile time, maybe increases perfomance
                return _{{ name }}(
            {%- for symbol in allSymbols %}
                    {{ symbol }},
            {%- endfor %}
                )
            
            cdef double complex _{{ name }}(
            {%- for symbol in allSymbols %}
                double complex {{ symbol }},
            {%- endfor %}
                ):
                cdef double complex a = 0.0
            {%- for op, term in opsAndExpressions %}
                a {{ op }} {{ term }}
            {%- endfor %}
                return a
            """)).render(name=name, allSymbols=allSymbols, opsAndExpressions=np.transpose(mutliLineExpression(veffFp))))


def generateDiagonalizeSubModule(
    moduleName, 
    allSymbols, 
    scalarMassMatrixFile, 
    scalarMassNames,
):
    with open(scalarMassMatrixFile) as file:
        scalarMassMatrices = [convertMatrixToCythonSyntax(line) for line in file.readlines()]

    # Creates a cython module with that computes an order of Veff
    with open(moduleName, 'w') as file:
    
        file.write(Environment().from_string(dedent("""\

            from  scipy.linalg import lapack
            
            cpdef double complex eigen(
            {%- for symbol in allSymbols %}
                double complex {{ symbol }},
            {%- endfor %}
            ):
                
                return _eigen(
            {%- for symbol in allSymbols %}
                    {{ symbol }},
            {%- endfor %}
                )
            
            cdef double complex _eigen(
            {%- for symbol in allSymbols %}
                double complex {{ symbol }},
            {%- endfor %}
            ):

            {%- for scalarMassMatrix in scalarMassMatrices %}
                scalarMassMatrix{{ loop.index0}} = {{ scalarMassMatrix }}
                eigenValues{{ loop.index0 }}, eigenVectors{{ loop.index0 }}, _ = lapack.dsyevd(scalarMassMatrix{{ loop.index0 }}, compute_v = 1)
            {%- endfor %}

            {% set scalarMassMatrixLength = (scalarMassNames | length) / (scalarMassMatrices | length) | int %}
            {%- for massSymbol in scalarMassNames %}
                {{ massSymbol }} = eigenValues{{ (loop.index0 / scalarMassMatrixLength) | int }}[{{ (loop.index0 % scalarMassMatrixLength) | int }}]
            {%- endfor %}
            """)).render(
                allSymbols=allSymbols, 
                scalarMassMatrices = scalarMassMatrices,
                scalarMassNames = scalarMassNames,
            ))


def mutliLineExpression(filePointer):
    ## Takes an expressions and breaks it down into a mutli line expression
    ## (Cython seems to struggle with the one line NNLO veff)
    
    with open(filePointer, 'r') as file:
        veff = file.read()
    
    operations = ["+="]
    expressions = []
    
    netBrackets = 0
    start = 0
    
    for i, char in enumerate(veff):
        if char == '(':
            netBrackets += 1
        elif char == ')':
            netBrackets -= 1
        if char == ' ' and netBrackets == 0:
            ##+1 to catch space
            line = veff[start:i+1]
            if line in ["+ ", "- "]:
                operations.append("+=" if line == "+ " else "-=")
            else:
                expressions.append(convertToCythonSyntax(line))
            start = i + 1
    
    # Any remaining characters should just be expressions
    if start < len(veff):
        line = veff[start:]
        expressions.append(convertToCythonSyntax(line))
    return operations, expressions

def convertMatrixToCythonSyntax(term):
    term = convertToCythonSyntax(term)
    term = term.replace('{', '[')
    return term.replace('}', ']')
    
def convertToCythonSyntax(term):
    term = term.replace('Sqrt', 'csqrt')
    term = term.replace('Log', 'clog')
    term = term.replace('[', '(')
    term = term.replace(']', ')')
    term = term.replace('^', '**')
    term = PythoniseMathematica.replaceSymbolsConst(term)
    return PythoniseMathematica.replaceGreekSymbols(term)
