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
        
    opsAndTerms = np.transpose(read_and_parse_terms(veffFp))
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
            {%- for op, term in opsAndTerms %}
                a {{ op }} {{ term }}
            {%- endfor %}
                return a
            """)).render(name=name, allSymbols=allSymbols, opsAndTerms=opsAndTerms))


def convert_to_cython_syntax(term):
    term = term.replace('Sqrt', 'csqrt')
    term = term.replace('Log', 'clog')
    term = term.replace('[', '(')
    term = term.replace(']', ')')
    term = term.replace('^', '**')
    term = PythoniseMathematica.replaceSymbolsConst(term)
    return PythoniseMathematica.replaceGreekSymbols(term)


def mutliLineExpression(filename):
    ## Takes an expressions and breaks it down into a mutli line expression
    ## (Cython seems to struggle with the one line NNLO veff)
    num_unpaired_brackets = 0
    next_line = []
    terms = []
    operations = ["+="]
    
    with open(filename, 'r') as file:
        while True:
            char = file.read(1)
            if not char:
                terms.append(convert_to_cython_syntax(''.join(next_line)))
                break

            next_line.append(char)

            if char == '(':
                num_unpaired_brackets += 1
            elif char == ')':
                num_unpaired_brackets -= 1

            if char == ' ' and num_unpaired_brackets == 0:
                line = ''.join(next_line)
                if line in ["+ ", "- "]:
                    operations.append("+=" if line == "+ " else "-=")
                else:
                    terms.append(convert_to_cython_syntax(line))
    
                next_line = []
    
    return operations, terms
