’ó
æ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8üÜ
²
'le_net5/feature_extractor/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'le_net5/feature_extractor/conv2d/kernel
«
;le_net5/feature_extractor/conv2d/kernel/Read/ReadVariableOpReadVariableOp'le_net5/feature_extractor/conv2d/kernel*&
_output_shapes
:*
dtype0
¢
%le_net5/feature_extractor/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%le_net5/feature_extractor/conv2d/bias

9le_net5/feature_extractor/conv2d/bias/Read/ReadVariableOpReadVariableOp%le_net5/feature_extractor/conv2d/bias*
_output_shapes
:*
dtype0
¶
)le_net5/feature_extractor/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)le_net5/feature_extractor/conv2d_1/kernel
Æ
=le_net5/feature_extractor/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp)le_net5/feature_extractor/conv2d_1/kernel*&
_output_shapes
:*
dtype0
¦
'le_net5/feature_extractor/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'le_net5/feature_extractor/conv2d_1/bias

;le_net5/feature_extractor/conv2d_1/bias/Read/ReadVariableOpReadVariableOp'le_net5/feature_extractor/conv2d_1/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	x*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	x*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:x*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:x*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:xT*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:xT*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:T*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:T*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:T
*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:T
*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Č)
value¾)B») B“)

zero_padding
feature_extractor

classifier
regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
	regularization_losses

trainable_variables
	variables
	keras_api

	conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
trainable_variables
	variables
	keras_api
Ō
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
 
F
0
1
2
 3
!4
"5
#6
$7
%8
&9
F
0
1
2
 3
!4
"5
#6
$7
%8
&9
­
regularization_losses

'layers
(non_trainable_variables
trainable_variables
)metrics
*layer_metrics
+layer_regularization_losses
	variables
 
 
 
 
­
	regularization_losses

,layers
-non_trainable_variables

trainable_variables
.metrics
/layer_metrics
0layer_regularization_losses
	variables
h

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
R
5regularization_losses
6trainable_variables
7	variables
8	keras_api
h

kernel
 bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
R
=regularization_losses
>trainable_variables
?	variables
@	keras_api
 

0
1
2
 3

0
1
2
 3
­
regularization_losses

Alayers
Bnon_trainable_variables
trainable_variables
Cmetrics
Dlayer_metrics
Elayer_regularization_losses
	variables
{
F_inbound_nodes
G_outbound_nodes
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api

L_inbound_nodes

!kernel
"bias
M_outbound_nodes
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api

R_inbound_nodes

#kernel
$bias
S_outbound_nodes
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
|
X_inbound_nodes

%kernel
&bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
 
*
!0
"1
#2
$3
%4
&5
*
!0
"1
#2
$3
%4
&5
­
regularization_losses

]layers
^non_trainable_variables
trainable_variables
_metrics
`layer_metrics
alayer_regularization_losses
	variables
mk
VARIABLE_VALUE'le_net5/feature_extractor/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%le_net5/feature_extractor/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)le_net5/feature_extractor/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'le_net5/feature_extractor/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_2/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_2/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 
 
 
 
 
 
 
 

0
1

0
1
­
1regularization_losses

blayers
cnon_trainable_variables
2trainable_variables
dmetrics
elayer_metrics
flayer_regularization_losses
3	variables
 
 
 
­
5regularization_losses

glayers
hnon_trainable_variables
6trainable_variables
imetrics
jlayer_metrics
klayer_regularization_losses
7	variables
 

0
 1

0
 1
­
9regularization_losses

llayers
mnon_trainable_variables
:trainable_variables
nmetrics
olayer_metrics
player_regularization_losses
;	variables
 
 
 
­
=regularization_losses

qlayers
rnon_trainable_variables
>trainable_variables
smetrics
tlayer_metrics
ulayer_regularization_losses
?	variables

0
1
2
3
 
 
 
 
 
 
 
 
 
­
Hregularization_losses

vlayers
wnon_trainable_variables
Itrainable_variables
xmetrics
ylayer_metrics
zlayer_regularization_losses
J	variables
 
 
 

!0
"1

!0
"1
­
Nregularization_losses

{layers
|non_trainable_variables
Otrainable_variables
}metrics
~layer_metrics
layer_regularization_losses
P	variables
 
 
 

#0
$1

#0
$1
²
Tregularization_losses
layers
non_trainable_variables
Utrainable_variables
metrics
layer_metrics
 layer_regularization_losses
V	variables
 
 

%0
&1

%0
&1
²
Yregularization_losses
layers
non_trainable_variables
Ztrainable_variables
metrics
layer_metrics
 layer_regularization_losses
[	variables

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

serving_default_input_1Placeholder*/
_output_shapes
:’’’’’’’’’*
dtype0*$
shape:’’’’’’’’’
Į
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1'le_net5/feature_extractor/conv2d/kernel%le_net5/feature_extractor/conv2d/bias)le_net5/feature_extractor/conv2d_1/kernel'le_net5/feature_extractor/conv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_4445275
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
č
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename;le_net5/feature_extractor/conv2d/kernel/Read/ReadVariableOp9le_net5/feature_extractor/conv2d/bias/Read/ReadVariableOp=le_net5/feature_extractor/conv2d_1/kernel/Read/ReadVariableOp;le_net5/feature_extractor/conv2d_1/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_4445888

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename'le_net5/feature_extractor/conv2d/kernel%le_net5/feature_extractor/conv2d/bias)le_net5/feature_extractor/conv2d_1/kernel'le_net5/feature_extractor/conv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_4445928ü
ü

N__inference_feature_extractor_layer_call_and_return_conditional_losses_4445071
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp“
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
conv2d/TanhČ
average_pooling2d/AvgPoolAvgPoolconv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpŪ
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2
conv2d_1/TanhĪ
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool
IdentityIdentity$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’  :::::R N
/
_output_shapes
:’’’’’’’’’  

_user_specified_namex
å
g
K__inference_zero_padding2d_layer_call_and_return_conditional_losses_4444821

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs

Ä
G__inference_sequential_layer_call_and_return_conditional_losses_4445031

inputs
dense_4445015
dense_4445017
dense_1_4445020
dense_1_4445022
dense_2_4445025
dense_2_4445027
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÓ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_44448612
flatten/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4445015dense_4445017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44448802
dense/StatefulPartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4445020dense_1_4445022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_44449072!
dense_1/StatefulPartitionedCall“
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4445025dense_2_4445027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_44449342!
dense_2/StatefulPartitionedCallą
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_4444861

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ž
ń
)__inference_le_net5_layer_call_fn_4445555
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_le_net5_layer_call_and_return_conditional_losses_44452002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:’’’’’’’’’

_user_specified_namex
ó
½
,__inference_sequential_layer_call_fn_4445659

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_44449942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

j
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_4444833

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
š
÷
)__inference_le_net5_layer_call_fn_4445415
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_le_net5_layer_call_and_return_conditional_losses_44452002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
@
Õ
"__inference__wrapped_model_4444814
input_1C
?le_net5_feature_extractor_conv2d_conv2d_readvariableop_resourceD
@le_net5_feature_extractor_conv2d_biasadd_readvariableop_resourceE
Ale_net5_feature_extractor_conv2d_1_conv2d_readvariableop_resourceF
Ble_net5_feature_extractor_conv2d_1_biasadd_readvariableop_resource;
7le_net5_sequential_dense_matmul_readvariableop_resource<
8le_net5_sequential_dense_biasadd_readvariableop_resource=
9le_net5_sequential_dense_1_matmul_readvariableop_resource>
:le_net5_sequential_dense_1_biasadd_readvariableop_resource=
9le_net5_sequential_dense_2_matmul_readvariableop_resource>
:le_net5_sequential_dense_2_biasadd_readvariableop_resource
identity»
#le_net5/zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2%
#le_net5/zero_padding2d/Pad/paddings°
le_net5/zero_padding2d/PadPadinput_1,le_net5/zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’  2
le_net5/zero_padding2d/Padų
6le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp?le_net5_feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOp¤
'le_net5/feature_extractor/conv2d/Conv2DConv2D#le_net5/zero_padding2d/Pad:output:0>le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2)
'le_net5/feature_extractor/conv2d/Conv2Dļ
7le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp@le_net5_feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOp
(le_net5/feature_extractor/conv2d/BiasAddBiasAdd0le_net5/feature_extractor/conv2d/Conv2D:output:0?le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2*
(le_net5/feature_extractor/conv2d/BiasAddĆ
%le_net5/feature_extractor/conv2d/TanhTanh1le_net5/feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2'
%le_net5/feature_extractor/conv2d/Tanh
3le_net5/feature_extractor/average_pooling2d/AvgPoolAvgPool)le_net5/feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
25
3le_net5/feature_extractor/average_pooling2d/AvgPoolž
8le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAle_net5_feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOpĆ
)le_net5/feature_extractor/conv2d_1/Conv2DConv2D<le_net5/feature_extractor/average_pooling2d/AvgPool:output:0@le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2+
)le_net5/feature_extractor/conv2d_1/Conv2Dõ
9le_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBle_net5_feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9le_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOp
*le_net5/feature_extractor/conv2d_1/BiasAddBiasAdd2le_net5/feature_extractor/conv2d_1/Conv2D:output:0Ale_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2,
*le_net5/feature_extractor/conv2d_1/BiasAddÉ
'le_net5/feature_extractor/conv2d_1/TanhTanh3le_net5/feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2)
'le_net5/feature_extractor/conv2d_1/Tanh
5le_net5/feature_extractor/average_pooling2d_1/AvgPoolAvgPool+le_net5/feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
27
5le_net5/feature_extractor/average_pooling2d_1/AvgPool
 le_net5/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2"
 le_net5/sequential/flatten/Constń
"le_net5/sequential/flatten/ReshapeReshape>le_net5/feature_extractor/average_pooling2d_1/AvgPool:output:0)le_net5/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2$
"le_net5/sequential/flatten/ReshapeŁ
.le_net5/sequential/dense/MatMul/ReadVariableOpReadVariableOp7le_net5_sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype020
.le_net5/sequential/dense/MatMul/ReadVariableOpć
le_net5/sequential/dense/MatMulMatMul+le_net5/sequential/flatten/Reshape:output:06le_net5/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2!
le_net5/sequential/dense/MatMul×
/le_net5/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp8le_net5_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype021
/le_net5/sequential/dense/BiasAdd/ReadVariableOpå
 le_net5/sequential/dense/BiasAddBiasAdd)le_net5/sequential/dense/MatMul:product:07le_net5/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2"
 le_net5/sequential/dense/BiasAdd£
le_net5/sequential/dense/TanhTanh)le_net5/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
le_net5/sequential/dense/TanhŽ
0le_net5/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp9le_net5_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype022
0le_net5/sequential/dense_1/MatMul/ReadVariableOpß
!le_net5/sequential/dense_1/MatMulMatMul!le_net5/sequential/dense/Tanh:y:08le_net5/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2#
!le_net5/sequential/dense_1/MatMulŻ
1le_net5/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp:le_net5_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype023
1le_net5/sequential/dense_1/BiasAdd/ReadVariableOpķ
"le_net5/sequential/dense_1/BiasAddBiasAdd+le_net5/sequential/dense_1/MatMul:product:09le_net5/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2$
"le_net5/sequential/dense_1/BiasAdd©
le_net5/sequential/dense_1/TanhTanh+le_net5/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2!
le_net5/sequential/dense_1/TanhŽ
0le_net5/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp9le_net5_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype022
0le_net5/sequential/dense_2/MatMul/ReadVariableOpį
!le_net5/sequential/dense_2/MatMulMatMul#le_net5/sequential/dense_1/Tanh:y:08le_net5/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2#
!le_net5/sequential/dense_2/MatMulŻ
1le_net5/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp:le_net5_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1le_net5/sequential/dense_2/BiasAdd/ReadVariableOpķ
"le_net5/sequential/dense_2/BiasAddBiasAdd+le_net5/sequential/dense_2/MatMul:product:09le_net5/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2$
"le_net5/sequential/dense_2/BiasAdd²
"le_net5/sequential/dense_2/SoftmaxSoftmax+le_net5/sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2$
"le_net5/sequential/dense_2/Softmax
IdentityIdentity,le_net5/sequential/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’:::::::::::X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1

¬
D__inference_dense_1_layer_call_and_return_conditional_losses_4445806

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’T2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’x:::O K
'
_output_shapes
:’’’’’’’’’x
 
_user_specified_nameinputs
Ü
~
)__inference_dense_2_layer_call_fn_4445835

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_44449342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’T::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’T
 
_user_specified_nameinputs
³
O
3__inference_average_pooling2d_layer_call_fn_4444839

inputs
identityļ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_44448332
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
Ź
ó
%__inference_signature_wrapper_4445275
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_44448142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
Ą.
ų
#__inference__traced_restore_4445928
file_prefix<
8assignvariableop_le_net5_feature_extractor_conv2d_kernel<
8assignvariableop_1_le_net5_feature_extractor_conv2d_bias@
<assignvariableop_2_le_net5_feature_extractor_conv2d_1_kernel>
:assignvariableop_3_le_net5_feature_extractor_conv2d_1_bias#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesā
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity·
AssignVariableOpAssignVariableOp8assignvariableop_le_net5_feature_extractor_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1½
AssignVariableOp_1AssignVariableOp8assignvariableop_1_le_net5_feature_extractor_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Į
AssignVariableOp_2AssignVariableOp<assignvariableop_2_le_net5_feature_extractor_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3æ
AssignVariableOp_3AssignVariableOp:assignvariableop_3_le_net5_feature_extractor_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¤
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¦
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¤
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpŗ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

l
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_4444845

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
 #

 __inference__traced_save_4445888
file_prefixF
Bsavev2_le_net5_feature_extractor_conv2d_kernel_read_readvariableopD
@savev2_le_net5_feature_extractor_conv2d_bias_read_readvariableopH
Dsavev2_le_net5_feature_extractor_conv2d_1_kernel_read_readvariableopF
Bsavev2_le_net5_feature_extractor_conv2d_1_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8d846bb87a064c128326ae0a983041bd/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueBB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesĢ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_le_net5_feature_extractor_conv2d_kernel_read_readvariableop@savev2_le_net5_feature_extractor_conv2d_bias_read_readvariableopDsavev2_le_net5_feature_extractor_conv2d_1_kernel_read_readvariableopBsavev2_le_net5_feature_extractor_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ŗ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes”
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*x
_input_shapesg
e: :::::	x:x:xT:T:T
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	x: 

_output_shapes
:x:$ 

_output_shapes

:xT: 

_output_shapes
:T:$	 

_output_shapes

:T
: 


_output_shapes
:
:

_output_shapes
: 

Ä
,__inference_sequential_layer_call_fn_4445747
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_44449942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameflatten_input

Ä
G__inference_sequential_layer_call_and_return_conditional_losses_4444994

inputs
dense_4444978
dense_4444980
dense_1_4444983
dense_1_4444985
dense_2_4444988
dense_2_4444990
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallÓ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_44448612
flatten/PartitionedCall¢
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_4444978dense_4444980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44448802
dense/StatefulPartitionedCall²
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_4444983dense_1_4444985*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_44449072!
dense_1/StatefulPartitionedCall“
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_4444988dense_2_4444990*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_44449342!
dense_2/StatefulPartitionedCallą
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ø9
§
D__inference_le_net5_layer_call_and_return_conditional_losses_4445365
input_1;
7feature_extractor_conv2d_conv2d_readvariableop_resource<
8feature_extractor_conv2d_biasadd_readvariableop_resource=
9feature_extractor_conv2d_1_conv2d_readvariableop_resource>
:feature_extractor_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings
zero_padding2d/PadPadinput_1$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’  2
zero_padding2d/Padą
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOp
feature_extractor/conv2d/Conv2DConv2Dzero_padding2d/Pad:output:06feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D×
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpģ
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2"
 feature_extractor/conv2d/BiasAdd«
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
feature_extractor/conv2d/Tanhž
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolę
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DŻ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpō
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2!
feature_extractor/conv2d_1/Tanh
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
sequential/flatten/ConstŃ
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/flatten/ReshapeĮ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĆ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/BiasAdd
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/TanhĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpæ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/BiasAdd
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/TanhĘ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpĮ
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/MatMulÅ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpĶ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/BiasAdd
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/Softmaxx
IdentityIdentity$sequential/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’:::::::::::X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1

ō
G__inference_sequential_layer_call_and_return_conditional_losses_4445730
flatten_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2

dense/Tanh„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/Tanh„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::^ Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameflatten_input

Ä
,__inference_sequential_layer_call_fn_4445764
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_44450312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameflatten_input
ø9
§
D__inference_le_net5_layer_call_and_return_conditional_losses_4445320
input_1;
7feature_extractor_conv2d_conv2d_readvariableop_resource<
8feature_extractor_conv2d_biasadd_readvariableop_resource=
9feature_extractor_conv2d_1_conv2d_readvariableop_resource>
:feature_extractor_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings
zero_padding2d/PadPadinput_1$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’  2
zero_padding2d/Padą
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOp
feature_extractor/conv2d/Conv2DConv2Dzero_padding2d/Pad:output:06feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D×
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpģ
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2"
 feature_extractor/conv2d/BiasAdd«
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
feature_extractor/conv2d/Tanhž
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolę
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DŻ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpō
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2!
feature_extractor/conv2d_1/Tanh
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
sequential/flatten/ConstŃ
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/flatten/ReshapeĮ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĆ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/BiasAdd
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/TanhĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpæ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/BiasAdd
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/TanhĘ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpĮ
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/MatMulÅ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpĶ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/BiasAdd
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/Softmaxx
IdentityIdentity$sequential/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’:::::::::::X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
š
÷
)__inference_le_net5_layer_call_fn_4445390
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCallŻ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_le_net5_layer_call_and_return_conditional_losses_44452002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:’’’’’’’’’
!
_user_specified_name	input_1
 
Ŗ
B__inference_dense_layer_call_and_return_conditional_losses_4445786

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’x2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¦9
”
D__inference_le_net5_layer_call_and_return_conditional_losses_4445505
x;
7feature_extractor_conv2d_conv2d_readvariableop_resource<
8feature_extractor_conv2d_biasadd_readvariableop_resource=
9feature_extractor_conv2d_1_conv2d_readvariableop_resource>
:feature_extractor_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings
zero_padding2d/PadPadx$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’  2
zero_padding2d/Padą
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOp
feature_extractor/conv2d/Conv2DConv2Dzero_padding2d/Pad:output:06feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D×
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpģ
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2"
 feature_extractor/conv2d/BiasAdd«
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
feature_extractor/conv2d/Tanhž
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolę
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DŻ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpō
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2!
feature_extractor/conv2d_1/Tanh
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
sequential/flatten/ConstŃ
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/flatten/ReshapeĮ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĆ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/BiasAdd
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/TanhĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpæ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/BiasAdd
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/TanhĘ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpĮ
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/MatMulÅ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpĶ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/BiasAdd
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/Softmaxx
IdentityIdentity$sequential/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’:::::::::::R N
/
_output_shapes
:’’’’’’’’’

_user_specified_namex

¬
D__inference_dense_1_layer_call_and_return_conditional_losses_4444907

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’T2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’x:::O K
'
_output_shapes
:’’’’’’’’’x
 
_user_specified_nameinputs
ś
ķ
G__inference_sequential_layer_call_and_return_conditional_losses_4445615

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2

dense/Tanh„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/Tanh„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
ś
ķ
G__inference_sequential_layer_call_and_return_conditional_losses_4445642

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2

dense/Tanh„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/Tanh„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
Ś
|
'__inference_dense_layer_call_fn_4445795

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallņ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’x*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_44448802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’x2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
½
»
D__inference_le_net5_layer_call_and_return_conditional_losses_4445200
x
feature_extractor_4445177
feature_extractor_4445179
feature_extractor_4445181
feature_extractor_4445183
sequential_4445186
sequential_4445188
sequential_4445190
sequential_4445192
sequential_4445194
sequential_4445196
identity¢)feature_extractor/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallź
zero_padding2d/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’  * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_zero_padding2d_layer_call_and_return_conditional_losses_44448212 
zero_padding2d/PartitionedCall§
)feature_extractor/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0feature_extractor_4445177feature_extractor_4445179feature_extractor_4445181feature_extractor_4445183*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_feature_extractor_layer_call_and_return_conditional_losses_44450712+
)feature_extractor/StatefulPartitionedCall„
"sequential/StatefulPartitionedCallStatefulPartitionedCall2feature_extractor/StatefulPartitionedCall:output:0sequential_4445186sequential_4445188sequential_4445190sequential_4445192sequential_4445194sequential_4445196*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_44450312$
"sequential/StatefulPartitionedCallŠ
IdentityIdentity+sequential/StatefulPartitionedCall:output:0*^feature_extractor/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::2V
)feature_extractor/StatefulPartitionedCall)feature_extractor/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ü
~
)__inference_dense_1_layer_call_fn_4445815

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallō
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_44449072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’T2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’x::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:’’’’’’’’’x
 
_user_specified_nameinputs
·
Q
5__inference_average_pooling2d_1_layer_call_fn_4444851

inputs
identityń
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_44448452
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
±
¬
D__inference_dense_2_layer_call_and_return_conditional_losses_4444934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’T:::O K
'
_output_shapes
:’’’’’’’’’T
 
_user_specified_nameinputs
¦9
”
D__inference_le_net5_layer_call_and_return_conditional_losses_4445460
x;
7feature_extractor_conv2d_conv2d_readvariableop_resource<
8feature_extractor_conv2d_biasadd_readvariableop_resource=
9feature_extractor_conv2d_1_conv2d_readvariableop_resource>
:feature_extractor_conv2d_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity«
zero_padding2d/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d/Pad/paddings
zero_padding2d/PadPadx$zero_padding2d/Pad/paddings:output:0*
T0*/
_output_shapes
:’’’’’’’’’  2
zero_padding2d/Padą
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOp
feature_extractor/conv2d/Conv2DConv2Dzero_padding2d/Pad:output:06feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D×
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpģ
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2"
 feature_extractor/conv2d/BiasAdd«
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
feature_extractor/conv2d/Tanhž
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolę
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DŻ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpō
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2!
feature_extractor/conv2d_1/Tanh
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPool
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
sequential/flatten/ConstŃ
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
sequential/flatten/ReshapeĮ
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02(
&sequential/dense/MatMul/ReadVariableOpĆ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/MatMulæ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÅ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/BiasAdd
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
sequential/dense/TanhĘ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpæ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/MatMulÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpĶ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/BiasAdd
sequential/dense_1/TanhTanh#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
sequential/dense_1/TanhĘ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOpĮ
sequential/dense_2/MatMulMatMulsequential/dense_1/Tanh:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/MatMulÅ
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpĶ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/BiasAdd
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
sequential/dense_2/Softmaxx
IdentityIdentity$sequential/dense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’:::::::::::R N
/
_output_shapes
:’’’’’’’’’

_user_specified_namex
Ž
ń
)__inference_le_net5_layer_call_fn_4445530
x
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_le_net5_layer_call_and_return_conditional_losses_44452002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:’’’’’’’’’::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:’’’’’’’’’

_user_specified_namex
ó
½
,__inference_sequential_layer_call_fn_4445676

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:’’’’’’’’’
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_44450312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
±
¬
D__inference_dense_2_layer_call_and_return_conditional_losses_4445826

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’T:::O K
'
_output_shapes
:’’’’’’’’’T
 
_user_specified_nameinputs
­
L
0__inference_zero_padding2d_layer_call_fn_4444827

inputs
identityģ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_zero_padding2d_layer_call_and_return_conditional_losses_44448212
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’:r n
J
_output_shapes8
6:4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 
_user_specified_nameinputs
ü

N__inference_feature_extractor_layer_call_and_return_conditional_losses_4445575
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityŖ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp“
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’*
paddingVALID*
strides
2
conv2d/Conv2D”
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp¤
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’2
conv2d/TanhČ
average_pooling2d/AvgPoolAvgPoolconv2d/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpŪ
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

*
paddingVALID*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:’’’’’’’’’

2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:’’’’’’’’’

2
conv2d_1/TanhĪ
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Tanh:y:0*
T0*/
_output_shapes
:’’’’’’’’’*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool
IdentityIdentity$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’  :::::R N
/
_output_shapes
:’’’’’’’’’  

_user_specified_namex
¤
E
)__inference_flatten_layer_call_fn_4445775

inputs
identityĆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_44448612
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
¼
`
D__inference_flatten_layer_call_and_return_conditional_losses_4445770

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:’’’’’’’’’2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*.
_input_shapes
:’’’’’’’’’:W S
/
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs
 
Ŗ
B__inference_dense_layer_call_and_return_conditional_losses_4444880

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:’’’’’’’’’x2

Identity"
identityIdentity:output:0*/
_input_shapes
:’’’’’’’’’:::P L
(
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs

ō
G__inference_sequential_layer_call_and_return_conditional_losses_4445703
flatten_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"’’’’  2
flatten/Const
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:’’’’’’’’’2
flatten/Reshape 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	x*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:x*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’x2
dense/BiasAddj

dense/TanhTanhdense/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’x2

dense/Tanh„
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:xT*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/MatMul¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp”
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/BiasAddp
dense_1/TanhTanhdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’T2
dense_1/Tanh„
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Tanh:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/MatMul¤
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp”
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/BiasAddy
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:’’’’’’’’’
2
dense_2/Softmaxm
IdentityIdentitydense_2/Softmax:softmax:0*
T0*'
_output_shapes
:’’’’’’’’’
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:’’’’’’’’’:::::::^ Z
/
_output_shapes
:’’’’’’’’’
'
_user_specified_nameflatten_input
Ā
”
3__inference_feature_extractor_layer_call_fn_4445588
x
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:’’’’’’’’’*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_feature_extractor_layer_call_and_return_conditional_losses_44450712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:’’’’’’’’’2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:’’’’’’’’’  ::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:’’’’’’’’’  

_user_specified_namex"øL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*³
serving_default
C
input_18
serving_default_input_1:0’’’’’’’’’<
output_10
StatefulPartitionedCall:0’’’’’’’’’
tensorflow/serving/predict:Ō
ķ
zero_padding
feature_extractor

classifier
regularization_losses
trainable_variables
	variables
	keras_api

signatures
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"õ
_tf_keras_modelŪ{"class_name": "LeNet5", "name": "le_net5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNet5"}}

	regularization_losses

trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ö
_tf_keras_layerÜ{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ė
	conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"¤
_tf_keras_layer{"class_name": "FeatureExtractor", "name": "feature_extractor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_sequentialķ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 120, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 5, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 5, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 120, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
 "
trackable_list_wrapper
f
0
1
2
 3
!4
"5
#6
$7
%8
&9"
trackable_list_wrapper
f
0
1
2
 3
!4
"5
#6
$7
%8
&9"
trackable_list_wrapper
Ī
regularization_losses

'layers
(non_trainable_variables
trainable_variables
)metrics
*layer_metrics
+layer_regularization_losses
	variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
	regularization_losses

,layers
-non_trainable_variables

trainable_variables
.metrics
/layer_metrics
0layer_regularization_losses
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ģ	

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+&call_and_return_all_conditional_losses
__call__"Å
_tf_keras_layer«{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 1]}}

5regularization_losses
6trainable_variables
7	variables
8	keras_api
+&call_and_return_all_conditional_losses
__call__"ų
_tf_keras_layerŽ{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ń	

kernel
 bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+&call_and_return_all_conditional_losses
__call__"Ź
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 6]}}

=regularization_losses
>trainable_variables
?	variables
@	keras_api
+&call_and_return_all_conditional_losses
__call__"ü
_tf_keras_layerā{"class_name": "AveragePooling2D", "name": "average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
<
0
1
2
 3"
trackable_list_wrapper
°
regularization_losses

Alayers
Bnon_trainable_variables
trainable_variables
Cmetrics
Dlayer_metrics
Elayer_regularization_losses
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object

F_inbound_nodes
G_outbound_nodes
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+&call_and_return_all_conditional_losses
__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

L_inbound_nodes

!kernel
"bias
M_outbound_nodes
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
+&call_and_return_all_conditional_losses
__call__"Č
_tf_keras_layer®{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 120, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 400]}}

R_inbound_nodes

#kernel
$bias
S_outbound_nodes
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
+ &call_and_return_all_conditional_losses
”__call__"Ė
_tf_keras_layer±{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}

X_inbound_nodes

%kernel
&bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Ģ
_tf_keras_layer²{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 84]}}
 "
trackable_list_wrapper
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
J
!0
"1
#2
$3
%4
&5"
trackable_list_wrapper
°
regularization_losses

]layers
^non_trainable_variables
trainable_variables
_metrics
`layer_metrics
alayer_regularization_losses
	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
A:?2'le_net5/feature_extractor/conv2d/kernel
3:12%le_net5/feature_extractor/conv2d/bias
C:A2)le_net5/feature_extractor/conv2d_1/kernel
5:32'le_net5/feature_extractor/conv2d_1/bias
:	x2dense/kernel
:x2
dense/bias
 :xT2dense_1/kernel
:T2dense_1/bias
 :T
2dense_2/kernel
:
2dense_2/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°
1regularization_losses

blayers
cnon_trainable_variables
2trainable_variables
dmetrics
elayer_metrics
flayer_regularization_losses
3	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
5regularization_losses

glayers
hnon_trainable_variables
6trainable_variables
imetrics
jlayer_metrics
klayer_regularization_losses
7	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
°
9regularization_losses

llayers
mnon_trainable_variables
:trainable_variables
nmetrics
olayer_metrics
player_regularization_losses
;	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
=regularization_losses

qlayers
rnon_trainable_variables
>trainable_variables
smetrics
tlayer_metrics
ulayer_regularization_losses
?	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Hregularization_losses

vlayers
wnon_trainable_variables
Itrainable_variables
xmetrics
ylayer_metrics
zlayer_regularization_losses
J	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
°
Nregularization_losses

{layers
|non_trainable_variables
Otrainable_variables
}metrics
~layer_metrics
layer_regularization_losses
P	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
µ
Tregularization_losses
layers
non_trainable_variables
Utrainable_variables
metrics
layer_metrics
 layer_regularization_losses
V	variables
”__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
µ
Yregularization_losses
layers
non_trainable_variables
Ztrainable_variables
metrics
layer_metrics
 layer_regularization_losses
[	variables
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Ģ2É
D__inference_le_net5_layer_call_and_return_conditional_losses_4445365
D__inference_le_net5_layer_call_and_return_conditional_losses_4445460
D__inference_le_net5_layer_call_and_return_conditional_losses_4445320
D__inference_le_net5_layer_call_and_return_conditional_losses_4445505®
„²”
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ą2Ż
)__inference_le_net5_layer_call_fn_4445415
)__inference_le_net5_layer_call_fn_4445390
)__inference_le_net5_layer_call_fn_4445555
)__inference_le_net5_layer_call_fn_4445530®
„²”
FullArgSpec$
args
jself
jx

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
č2å
"__inference__wrapped_model_4444814¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *.¢+
)&
input_1’’’’’’’’’
³2°
K__inference_zero_padding2d_layer_call_and_return_conditional_losses_4444821ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
0__inference_zero_padding2d_layer_call_fn_4444827ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
ó2š
N__inference_feature_extractor_layer_call_and_return_conditional_losses_4445575
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ų2Õ
3__inference_feature_extractor_layer_call_fn_4445588
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ź2ē
G__inference_sequential_layer_call_and_return_conditional_losses_4445730
G__inference_sequential_layer_call_and_return_conditional_losses_4445615
G__inference_sequential_layer_call_and_return_conditional_losses_4445703
G__inference_sequential_layer_call_and_return_conditional_losses_4445642Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
ž2ū
,__inference_sequential_layer_call_fn_4445747
,__inference_sequential_layer_call_fn_4445676
,__inference_sequential_layer_call_fn_4445659
,__inference_sequential_layer_call_fn_4445764Ą
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŖ 
annotationsŖ *
 
4B2
%__inference_signature_wrapper_4445275input_1
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
¶2³
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_4444833ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
3__inference_average_pooling2d_layer_call_fn_4444839ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ø2„¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ø2µ
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_4444845ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
2
5__inference_average_pooling2d_1_layer_call_fn_4444851ą
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *@¢=
;84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
ī2ė
D__inference_flatten_layer_call_and_return_conditional_losses_4445770¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_flatten_layer_call_fn_4445775¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ģ2é
B__inference_dense_layer_call_and_return_conditional_losses_4445786¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ń2Ī
'__inference_dense_layer_call_fn_4445795¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_1_layer_call_and_return_conditional_losses_4445806¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_1_layer_call_fn_4445815¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
ī2ė
D__inference_dense_2_layer_call_and_return_conditional_losses_4445826¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 
Ó2Š
)__inference_dense_2_layer_call_fn_4445835¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 ”
"__inference__wrapped_model_4444814{
 !"#$%&8¢5
.¢+
)&
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’
ó
P__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_4444845R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ė
5__inference_average_pooling2d_1_layer_call_fn_4444851R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’ń
N__inference_average_pooling2d_layer_call_and_return_conditional_losses_4444833R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 É
3__inference_average_pooling2d_layer_call_fn_4444839R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’¤
D__inference_dense_1_layer_call_and_return_conditional_losses_4445806\#$/¢,
%¢"
 
inputs’’’’’’’’’x
Ŗ "%¢"

0’’’’’’’’’T
 |
)__inference_dense_1_layer_call_fn_4445815O#$/¢,
%¢"
 
inputs’’’’’’’’’x
Ŗ "’’’’’’’’’T¤
D__inference_dense_2_layer_call_and_return_conditional_losses_4445826\%&/¢,
%¢"
 
inputs’’’’’’’’’T
Ŗ "%¢"

0’’’’’’’’’

 |
)__inference_dense_2_layer_call_fn_4445835O%&/¢,
%¢"
 
inputs’’’’’’’’’T
Ŗ "’’’’’’’’’
£
B__inference_dense_layer_call_and_return_conditional_losses_4445786]!"0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’x
 {
'__inference_dense_layer_call_fn_4445795P!"0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’x»
N__inference_feature_extractor_layer_call_and_return_conditional_losses_4445575i 2¢/
(¢%
# 
x’’’’’’’’’  
Ŗ "-¢*
# 
0’’’’’’’’’
 
3__inference_feature_extractor_layer_call_fn_4445588\ 2¢/
(¢%
# 
x’’’’’’’’’  
Ŗ " ’’’’’’’’’©
D__inference_flatten_layer_call_and_return_conditional_losses_4445770a7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
)__inference_flatten_layer_call_fn_4445775T7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "’’’’’’’’’¹
D__inference_le_net5_layer_call_and_return_conditional_losses_4445320q
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’

 ¹
D__inference_le_net5_layer_call_and_return_conditional_losses_4445365q
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’

 ³
D__inference_le_net5_layer_call_and_return_conditional_losses_4445460k
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’

 ³
D__inference_le_net5_layer_call_and_return_conditional_losses_4445505k
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’

 
)__inference_le_net5_layer_call_fn_4445390d
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p
Ŗ "’’’’’’’’’

)__inference_le_net5_layer_call_fn_4445415d
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p 
Ŗ "’’’’’’’’’

)__inference_le_net5_layer_call_fn_4445530^
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p
Ŗ "’’’’’’’’’

)__inference_le_net5_layer_call_fn_4445555^
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p 
Ŗ "’’’’’’’’’
»
G__inference_sequential_layer_call_and_return_conditional_losses_4445615p!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 »
G__inference_sequential_layer_call_and_return_conditional_losses_4445642p!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 Ā
G__inference_sequential_layer_call_and_return_conditional_losses_4445703w!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p

 
Ŗ "%¢"

0’’’’’’’’’

 Ā
G__inference_sequential_layer_call_and_return_conditional_losses_4445730w!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p 

 
Ŗ "%¢"

0’’’’’’’’’

 
,__inference_sequential_layer_call_fn_4445659c!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

,__inference_sequential_layer_call_fn_4445676c!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

,__inference_sequential_layer_call_fn_4445747j!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

,__inference_sequential_layer_call_fn_4445764j!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
°
%__inference_signature_wrapper_4445275
 !"#$%&C¢@
¢ 
9Ŗ6
4
input_1)&
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’
ī
K__inference_zero_padding2d_layer_call_and_return_conditional_losses_4444821R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ę
0__inference_zero_padding2d_layer_call_fn_4444827R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’