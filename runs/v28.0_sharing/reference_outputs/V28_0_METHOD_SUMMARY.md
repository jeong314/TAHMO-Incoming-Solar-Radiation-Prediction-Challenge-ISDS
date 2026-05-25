# v28.0_asymmetric_edge_kt_tilt_reproduction_pack

## Hypothesis
Distance-weighted adjacent-month kt anomaly is applied as a mean-neutral shape correction to strong public/team anchors, especially jieun20. v28.0 separates early-edge and late-edge correction strength because validation and public feedback indicate an asymmetric seasonal energy tilt: early months need smaller correction than late months.

## Leakage guard
Validation months: [3, 5, 7, 9, 11]. Held-out target month energy is predicted from adjacent non-held-out odd months. Mean-neutral modes use predictions only, not held-out targets. Test candidates are emitted for jieun20/ds15 plus the validation anchor.

## Selected candidates
```csv
candidate,gate,neutral,dist_power,alpha_early,alpha_late,weighted,rmse,abs_mbe,risk_score,weighted_delta_vs_best_anchor,risk_delta_vs_best_anchor,submission_guard

shape_v177_add_b1.00_s0.30_ae0.18_al0.30_c0.16_dp0.75_edge_global,edge,global,0.75,0.18,0.3,34.89450175010169,69.46202113005049,0.32698237015288595,35.13261860403985,-0.0334073342312351,-0.06874177379746271,False

shape_v177_add_b1.00_s0.30_ae0.15_al0.15_c0.16_dp2.00_edge_global,edge,global,2.0,0.15,0.15,34.89108264960705,69.45518292906121,0.32698237015289067,35.13264492728915,-0.036826434725874435,-0.06871545054816153,False

shape_v177_add_b1.00_s0.30_ae0.15_al0.15_c0.16_dp1.25_edge_global,edge,global,1.25,0.15,0.15,34.89137062833807,69.45575888652326,0.32698237015289,35.13302744171972,-0.03653845599485095,-0.06833293611759217,False

shape_v177_add_b1.00_s0.30_ae0.18_al0.30_c0.16_dp1.25_edge_global,edge,global,1.25,0.18,0.3,34.894041609384686,69.46110084861648,0.3269823701528905,35.13326653055481,-0.03386747494823794,-0.06809384728249768,False

shape_v177_add_b1.00_s0.30_ae0.15_al0.15_c0.16_dp0.75_edge_global,edge,global,0.75,0.15,0.15,34.89163208584351,69.45628180153413,0.32698237015288856,35.13336602122556,-0.036276998489412904,-0.06799435661174869,False

shape_v177_add_b1.00_s0.30_ae0.15_al0.25_c0.16_dp0.75_edge_global,edge,global,0.75,0.15,0.25,34.89153344274829,69.45608451534369,0.32698237015288933,35.13376204372624,-0.03637564158463391,-0.06759833411106797,False

shape_v177_add_b1.00_s0.30_ae0.18_al0.30_c0.16_dp2.00_edge_global,edge,global,2.0,0.18,0.3,34.89358884927828,69.46019532840367,0.326982370152891,35.13418211227112,-0.03432023505464343,-0.06717826556619144,False

shape_v177_add_b1.00_s0.30_ae0.15_al0.25_c0.16_dp1.25_edge_global,edge,global,1.25,0.15,0.25,34.89114761467739,69.45531285920188,0.32698237015288684,35.134299605015315,-0.03676146965553784,-0.0670607728219963,False
```
