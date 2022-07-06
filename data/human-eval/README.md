# Human Evaluation Data

Responses from our human evaluation of various summary responses are available in `all-responses.jsonl`.
The file has the following format:

```
passage-id: the Gutenberg passage ID
questions: list in original question order, where each element contains
  |- {bart,bart+dpr,human}
     |- response: the original response
     |- reviews: list of review by reviewers
        |- annotations: list of annotations, where each annotation consists of
           |- selection: original response text highlighted by reviewer
           |- annotation: reviewer comment on the selection
        |- worker_id: the anonymous ID of the reviewer
        |- {correctness,selection,overall}-rating: rating of the system's response for that property for that reviewer
        |- {correctness,selection,overall}-rank: rank of the system's repsonse for that property for that reviewer
     |- overall
        |- {correctness,selection,overall}-rating: mean rating of the system's response for that property
        |- {correctness,selection,overall}-rank: overall rank of the system's repsonse for that property, based on overall ratings
    
```

For more details on the evluation was run, what instructions what propvided to reviewers, etc., please consult our paper and its appendix.
