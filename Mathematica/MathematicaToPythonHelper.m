(* ::Package:: *)

(************************************************************************)
(*
This is a collection of functions we use to manipulate mathematica
outputs into a form we can easily work with in BLOOP.
Most (all?) of the code here is written by claude v4
*)
(************************************************************************)



exportUTF8[fileName_, expr_] := Module[{},
  If[StringQ[expr],
    Export[fileName, expr, "Text", CharacterEncoding -> "UTF-8"],
    Export[fileName, expr, CharacterEncoding -> "UTF-8"]
  ]
];


(* ::Input::Initialization:: *)
(** Replace non-numeric matrix elements by simple symbols. eg. long expression at matrix[[i,j]] -> elementSymbolX
where X is a number starting from 0. Each element gets a unique symbol. Returns the symbolic matrix and list of shorthands. **)
toSymbolicMatrix[matrix_, elementSymbol_, bIsSymmetric_: False] := Block[
{i, j, count, symbolicMatrix, shape, rows, columns, shorthandDefinitions, tempMatrix},
	
	count = 0;
	shape = Dimensions[matrix];
	{rows, columns} = shape;
	
	(** don't replace numerical elements (esp. 0 or 1) **) 
	IsTrivialQ[el_] := Return[ NumericQ[el] ];
	
	(** Helper. SetAttributes is used in order to modify the shorthandList argument **)
	SetAttributes[AppendSymbolicShorthand, HoldAll];
	AppendSymbolicShorthand[el_, shorthandBase_, shorthandList_] := Block[{substRule, shorthand},
		If[ !IsTrivialQ[el],
			shorthand = ToExpression[ToString[shorthandBase]<>ToString[count]];
			substRule = shorthand -> el;
			AppendTo[shorthandList, substRule];
			count++;
			Return[shorthand];
			,
			(* else *)
			Return[el];
		];
	];
	
	shorthandDefinitions = {};
		
	If[!bIsSymmetric,
		symbolicMatrix = Table[ AppendSymbolicShorthand[ matrix[[i,j]], elementSymbol, shorthandDefinitions ], {i,1,rows},{j,1,columns}];
	,
	(* Symmetric matrix: fill in symbols in the upper right half only *)
		tempMatrix = Table[
			AppendSymbolicShorthand[ matrix[[i,j]], elementSymbol, shorthandDefinitions ],
		{i,1,rows},{j,1,i}];
		
		symbolicMatrix = Table[
			If[j<=i, tempMatrix[[i,j]], tempMatrix[[j,i]]], {i,1,rows}, {j,1,columns}
		];
	];
	
	Return[{symbolicMatrix, shorthandDefinitions}];
];


(* Written by Claude v4, bug fixed by ChatGPT o4-mini-high*)
extractSymbols[expr_, includeProtected_: False, asStrings_: True] :=
 Module[{syms, lhs, rhs, keepQ, toStr},

  (* keep or drop protected symbols *)
  keepQ = If[includeProtected, True &, ! MemberQ[Attributes[#], Protected] &];

  (* convert ONLY non-System symbols to strings *)
  toStr[s_] := If[asStrings, SymbolName[s], s];

  (* core extractor *)
  symsFrom[e_] :=
    DeleteDuplicates @ Select[Cases[e, _Symbol, Infinity], keepQ];

  Which[
   (* list of rules *)
   ListQ[expr] && AllTrue[expr, MatchQ[#, _Rule | _RuleDelayed] &],
   lhs = toStr /@ symsFrom[expr[[All, 1]]];
   rhs = toStr /@ symsFrom[expr[[All, 2]]];
   <|"LHS" -> lhs, "RHS" -> rhs|>,

   (* SparseArray *)
   Head[expr] === SparseArray,
   toStr /@ symsFrom[expr["NonzeroValues"]],

   True,
   toStr /@ symsFrom[expr]
  ]
]


symbolsFromDict[rules_List] :=
  DeleteDuplicates @ Flatten @ Cases[
    rules,
    r : (_Rule | _RuleDelayed) /; FreeQ[r[[2]], _Rule | _RuleDelayed] :> r[[2]],
    {0, \[Infinity]}
  ]


(*Claude 3.5*)
(* Helper function to remove any suffix from a symbol *)
RemoveSymbolSuffix[expr_Symbol, suffix_String] := 
  If[StringEndsQ[ToString[expr], suffix],
     Symbol[StringReplace[ToString[expr], suffix -> ""]],
     expr]

(* Main function to recursively process expressions *)
RemoveExpressionSuffix[expr_, suffix_String] := 
  expr /. s_Symbol :> RemoveSymbolSuffix[s, suffix]

(* Process entire rule list for a single suffix *)
RemoveSingleSuffix[rules_List, suffix_String] := 
  Map[Rule[
    RemoveSymbolSuffix[#[[1]], suffix], 
    RemoveExpressionSuffix[#[[2]], suffix]
  ] &, rules]

(* Process rules with multiple suffixes *)
RemoveSuffixes[rules_List, suffixes_List] := 
  Fold[RemoveSingleSuffix[#1, #2] &, rules, suffixes]


(**  Here the lists are assumed to have elements of form symbol -> expr. 
This function joins substitution rules from list1 and list2 so that the final subst rule is symbol -> expr1 + expr2. **)
CombineSubstRules[list1_, list2_] := Block[{combinedList,groupedRules},

	(** Magic code written by ChatGPT. But it works **)
	
	combinedList = Join[list1, list2];
	(* Group the rules by their left-hand sides *)
	groupedRules = GroupBy[combinedList, #[[1]] &];

	(* Sum up the right-hand sides for each group *)
	resultList = Rule @@@ KeyValueMap[{#1, Total[#2[[All, 2]]]} &, groupedRules];
	Return[resultList];
];


(*Gauge couplings given as g_i^2 even though g_i is what is needed, so take sqrt. Fine to do so long as g_i >0*)
sqrtSubRules[ruleList_]:=Module[{newRules},
  newRules = ruleList /. (lhs_ -> expr_) /; MatchQ[lhs, _^2] :> (PowerExpand[Sqrt[lhs]] -> Sqrt[expr]);
  Return[newRules];
];


(* Generate a json from a matrix that has elements as keys and 
index of the elements as value i.e. "ele": "idx1,idx2"*)
matrixToJSON[mat_] := ExportString[
  Association[ToString[#] -> (Position[mat, #][[1]] - 1) & /@ 
    Select[DeleteDuplicates[Cases[mat, _Symbol, All]], AtomQ[#] && # =!= List &]], 
  "JSON"
]



toIndexedSymbols[symbol_, ranges__] := 
  Module[{rangeList = List@ranges},
    Array[
      ToExpression[ToString[symbol] <> StringJoin[ToString /@ Flatten[{##}]]] &,
      rangeList[[All, 2]] - rangeList[[All, 1]] + 1,
      rangeList[[All, 1]]
    ]
  ];
