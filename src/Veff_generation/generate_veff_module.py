import os
from textwrap import dedent
from jinja2 import Environment
import numpy as np

import Bloop.PythoniseMathematica as PythoniseMathematica

def generate_veff_module(args, allSymbols):
    
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
            extensions.append(Extension("eignen", ["eigen.pyx"]))

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


def generateDiagonalizeSubModule(moduleName, allSymbols):
    # Creates a cython module with that computes an order of Veff
    with open(moduleName, 'w') as file:
    
        file.write(Environment().from_string(dedent("""\
            #cython: cdivision=False
            from libc.complex cimport csqrt
            from libc.complex cimport clog

            from  scipy.linalg import lapack
            
            def eigen(
            {%- for symbol in allSymbols %}
                double complex& {{ symbol }},
            {%- endfor %}
                ):
                scalarMassMatrix = [[lamda11*v1**2 + lamda12*v2**2/2 + lamda31*v3**2/2 - mu1sq, lamda3Im*v1*v3, 0, v1*v3*(lamda31p/2 + lamda3Re), -mu12sqRe + v1*v2*(lamda12p/2 + lamda1Re), -lamda1Im*v1*v2 + mu12sqIm], [lamda3Im*v1*v3, lamda23*v2**2/2 + lamda31*v1**2/2 + lamda33*v3**2 - mu3sq, v1*v3*(lamda31p/2 + lamda3Re), 0, -lamda2Im*v2*v3, v2*v3*(lamda23p/2 + lamda2Re)], [0, v1*v3*(lamda31p/2 + lamda3Re), lamda11*v1**2 + lamda12*v2**2/2 + lamda31*v3**2/2 - mu1sq, -lamda3Im*v1*v3, lamda1Im*v1*v2 - mu12sqIm, -mu12sqRe + v1*v2*(lamda12p/2 + lamda1Re)], [v1*v3*(lamda31p/2 + lamda3Re), 0, -lamda3Im*v1*v3, lamda23*v2**2/2 + lamda31*v1**2/2 + lamda33*v3**2 - mu3sq, v2*v3*(lamda23p/2 + lamda2Re), lamda2Im*v2*v3], [-mu12sqRe + v1*v2*(lamda12p/2 + lamda1Re), -lamda2Im*v2*v3, lamda1Im*v1*v2 - mu12sqIm, v2*v3*(lamda23p/2 + lamda2Re), lamda12*v1**2/2 + lamda22*v2**2 + lamda23*v3**2/2 - mu2sq, 0], [-lamda1Im*v1*v2 + mu12sqIm, v2*v3*(lamda23p/2 + lamda2Re), -mu12sqRe + v1*v2*(lamda12p/2 + lamda1Re), lamda2Im*v2*v3, 0, lamda12*v1**2/2 + lamda22*v2**2 + lamda23*v3**2/2 - mu2sq]]
                eigenValues, eigenVectors, _ = lapack.dsyevd(scalarMassMatrix, compute_v = 1)

                return eigenValues, eigenVectors
            """)).render(allSymbols=allSymbols))


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
                expressions.append(convert_to_cython_syntax(line))
            start = i + 1
    
    # Any remaining characters should just be expressions
    if start < len(veff):
        line = veff[start:]
        expressions.append(convert_to_cython_syntax(line))
    return operations, expressions
    
def convert_to_cython_syntax(term):
    term = term.replace('Sqrt', 'csqrt')
    term = term.replace('Log', 'clog')
    term = term.replace('[', '(')
    term = term.replace(']', ')')
    term = term.replace('^', '**')
    term = PythoniseMathematica.replaceSymbolsConst(term)
    return PythoniseMathematica.replaceGreekSymbols(term)
