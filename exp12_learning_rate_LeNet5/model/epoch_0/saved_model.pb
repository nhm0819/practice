š
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
 "serve*2.3.02unknown8ŁŁ
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
©)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ä(
valueŚ(B×( BŠ(

zero_padding
feature_extractor

classifier
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
	regularization_losses

	variables
trainable_variables
	keras_api

	conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
	variables
trainable_variables
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
	variables
trainable_variables
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

'layers
(layer_metrics
regularization_losses
	variables
trainable_variables
)layer_regularization_losses
*metrics
+non_trainable_variables
 
 
 
 
­

,layers
-layer_metrics
	regularization_losses

	variables
trainable_variables
.layer_regularization_losses
/metrics
0non_trainable_variables
h

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

kernel
 bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
R
=regularization_losses
>	variables
?trainable_variables
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

Alayers
Blayer_metrics
regularization_losses
	variables
trainable_variables
Clayer_regularization_losses
Dmetrics
Enon_trainable_variables
{
F_inbound_nodes
G_outbound_nodes
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api

L_inbound_nodes

!kernel
"bias
M_outbound_nodes
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api

R_inbound_nodes

#kernel
$bias
S_outbound_nodes
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
|
X_inbound_nodes

%kernel
&bias
Yregularization_losses
Z	variables
[trainable_variables
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

]layers
^layer_metrics
regularization_losses
	variables
trainable_variables
_layer_regularization_losses
`metrics
anon_trainable_variables
ca
VARIABLE_VALUE'le_net5/feature_extractor/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%le_net5/feature_extractor/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)le_net5/feature_extractor/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'le_net5/feature_extractor/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
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

blayers
clayer_metrics
1regularization_losses
2	variables
3trainable_variables
dlayer_regularization_losses
emetrics
fnon_trainable_variables
 
 
 
­

glayers
hlayer_metrics
5regularization_losses
6	variables
7trainable_variables
ilayer_regularization_losses
jmetrics
knon_trainable_variables
 

0
 1

0
 1
­

llayers
mlayer_metrics
9regularization_losses
:	variables
;trainable_variables
nlayer_regularization_losses
ometrics
pnon_trainable_variables
 
 
 
­

qlayers
rlayer_metrics
=regularization_losses
>	variables
?trainable_variables
slayer_regularization_losses
tmetrics
unon_trainable_variables

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

vlayers
wlayer_metrics
Hregularization_losses
I	variables
Jtrainable_variables
xlayer_regularization_losses
ymetrics
znon_trainable_variables
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

{layers
|layer_metrics
Nregularization_losses
O	variables
Ptrainable_variables
}layer_regularization_losses
~metrics
non_trainable_variables
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
layers
layer_metrics
Tregularization_losses
U	variables
Vtrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
 
 

%0
&1

%0
&1
²
layers
layer_metrics
Yregularization_losses
Z	variables
[trainable_variables
 layer_regularization_losses
metrics
non_trainable_variables

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
Ą
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
GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_278650
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ē
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
GPU 2J 8 *(
f#R!
__inference__traced_save_279263

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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_279303Ą
„
°
C__inference_le_net5_layer_call_and_return_conditional_losses_278575
x
feature_extractor_278552
feature_extractor_278554
feature_extractor_278556
feature_extractor_278558
sequential_278561
sequential_278563
sequential_278565
sequential_278567
sequential_278569
sequential_278571
identity¢)feature_extractor/StatefulPartitionedCall¢"sequential/StatefulPartitionedCallé
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
GPU 2J 8 *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2781962 
zero_padding2d/PartitionedCall¢
)feature_extractor/StatefulPartitionedCallStatefulPartitionedCall'zero_padding2d/PartitionedCall:output:0feature_extractor_278552feature_extractor_278554feature_extractor_278556feature_extractor_278558*
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
GPU 2J 8 *V
fQRO
M__inference_feature_extractor_layer_call_and_return_conditional_losses_2784462+
)feature_extractor/StatefulPartitionedCall
"sequential/StatefulPartitionedCallStatefulPartitionedCall2feature_extractor/StatefulPartitionedCall:output:0sequential_278561sequential_278563sequential_278565sequential_278567sequential_278569sequential_278571*
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
GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2784062$
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
Ś
}
(__inference_dense_2_layer_call_fn_279210

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2783092
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

«
C__inference_dense_1_layer_call_and_return_conditional_losses_279181

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
Ü
š
(__inference_le_net5_layer_call_fn_278905
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
identity¢StatefulPartitionedCallÖ
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
GPU 2J 8 *L
fGRE
C__inference_le_net5_layer_call_and_return_conditional_losses_2785752
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
ł
ģ
F__inference_sequential_layer_call_and_return_conditional_losses_279105

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
Ą
 
2__inference_feature_extractor_layer_call_fn_278963
x
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
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
GPU 2J 8 *V
fQRO
M__inference_feature_extractor_layer_call_and_return_conditional_losses_2784462
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
_user_specified_namex

k
O__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_278220

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

i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_278208

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
ū

M__inference_feature_extractor_layer_call_and_return_conditional_losses_278446
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
ń
¼
+__inference_sequential_layer_call_fn_279122

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallŖ
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
GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2783692
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

©
A__inference_dense_layer_call_and_return_conditional_losses_279161

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
@
Ō
!__inference__wrapped_model_278189
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
„9
 
C__inference_le_net5_layer_call_and_return_conditional_losses_278880
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
ń
¼
+__inference_sequential_layer_call_fn_279139

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallŖ
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
GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2784062
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

½
F__inference_sequential_layer_call_and_return_conditional_losses_278369

inputs
dense_278353
dense_278355
dense_1_278358
dense_1_278360
dense_2_278363
dense_2_278365
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallŅ
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
GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2782362
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_278353dense_278355*
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
GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2782552
dense/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_278358dense_1_278360*
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
GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2782822!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_278363dense_2_278365*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2783092!
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

«
C__inference_dense_1_layer_call_and_return_conditional_losses_278282

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
µ
P
4__inference_average_pooling2d_1_layer_call_fn_278226

inputs
identityš
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
GPU 2J 8 *X
fSRQ
O__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_2782202
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
»"

__inference__traced_save_279263
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
value3B1 B+_temp_0303e0e355ca4fcb9a71acf03488c0c8/part2	
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
ShardedFilename±
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć
value¹B¶B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
ī
ö
(__inference_le_net5_layer_call_fn_278765
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
identity¢StatefulPartitionedCallÜ
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
GPU 2J 8 *L
fGRE
C__inference_le_net5_layer_call_and_return_conditional_losses_2785752
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
Č
ņ
$__inference_signature_wrapper_278650
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
identity¢StatefulPartitionedCallŗ
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
GPU 2J 8 **
f%R#
!__inference__wrapped_model_2781892
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
ä
f
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_278196

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

½
F__inference_sequential_layer_call_and_return_conditional_losses_278406

inputs
dense_278390
dense_278392
dense_1_278395
dense_1_278397
dense_2_278400
dense_2_278402
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCallŅ
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
GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2782362
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_278390dense_278392*
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
GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2782552
dense/StatefulPartitionedCallÆ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_278395dense_1_278397*
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
GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2782822!
dense_1/StatefulPartitionedCall±
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_278400dense_2_278402*
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
GPU 2J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2783092!
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
°
«
C__inference_dense_2_layer_call_and_return_conditional_losses_278309

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
„9
 
C__inference_le_net5_layer_call_and_return_conditional_losses_278835
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
©
A__inference_dense_layer_call_and_return_conditional_losses_278255

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
·9
¦
C__inference_le_net5_layer_call_and_return_conditional_losses_278740
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

Ć
+__inference_sequential_layer_call_fn_279034
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall±
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
GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2783692
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
ł
ģ
F__inference_sequential_layer_call_and_return_conditional_losses_279078

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

ó
F__inference_sequential_layer_call_and_return_conditional_losses_279017
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
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_278236

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
°
«
C__inference_dense_2_layer_call_and_return_conditional_losses_279201

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
Ų
{
&__inference_dense_layer_call_fn_279170

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallń
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
GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2782552
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
Ś
}
(__inference_dense_1_layer_call_fn_279190

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
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
GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2782822
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
±
N
2__inference_average_pooling2d_layer_call_fn_278214

inputs
identityī
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
GPU 2J 8 *V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_2782082
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
ū

M__inference_feature_extractor_layer_call_and_return_conditional_losses_278950
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
Ü
š
(__inference_le_net5_layer_call_fn_278930
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
identity¢StatefulPartitionedCallÖ
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
GPU 2J 8 *L
fGRE
C__inference_le_net5_layer_call_and_return_conditional_losses_2785752
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
ī
ö
(__inference_le_net5_layer_call_fn_278790
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
identity¢StatefulPartitionedCallÜ
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
GPU 2J 8 *L
fGRE
C__inference_le_net5_layer_call_and_return_conditional_losses_2785752
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
«
K
/__inference_zero_padding2d_layer_call_fn_278202

inputs
identityė
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
GPU 2J 8 *S
fNRL
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_2781962
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
Ū-
÷
"__inference__traced_restore_279303
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
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9·
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ć
value¹B¶B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_279145

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
·9
¦
C__inference_le_net5_layer_call_and_return_conditional_losses_278695
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

ó
F__inference_sequential_layer_call_and_return_conditional_losses_278990
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

Ć
+__inference_sequential_layer_call_fn_279051
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall±
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
GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_2784062
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
¢
D
(__inference_flatten_layer_call_fn_279150

inputs
identityĀ
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
GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2782362
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
 
_user_specified_nameinputs"øL
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
tensorflow/serving/predict:
ķ
zero_padding
feature_extractor

classifier
regularization_losses
	variables
trainable_variables
	keras_api

signatures
__call__
_default_save_signature
+&call_and_return_all_conditional_losses"õ
_tf_keras_modelŪ{"class_name": "LeNet5", "name": "le_net5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNet5"}}

	regularization_losses

	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"ö
_tf_keras_layerÜ{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ė
	conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"¤
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
	variables
trainable_variables
	keras_api
__call__
+&call_and_return_all_conditional_losses"
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

'layers
(layer_metrics
regularization_losses
	variables
trainable_variables
)layer_regularization_losses
*metrics
+non_trainable_variables
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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

,layers
-layer_metrics
	regularization_losses

	variables
trainable_variables
.layer_regularization_losses
/metrics
0non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ģ	

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
__call__
+&call_and_return_all_conditional_losses"Å
_tf_keras_layer«{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 32, 32, 1]}}

5regularization_losses
6	variables
7trainable_variables
8	keras_api
__call__
+&call_and_return_all_conditional_losses"ų
_tf_keras_layerŽ{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ń	

kernel
 bias
9regularization_losses
:	variables
;trainable_variables
<	keras_api
__call__
+&call_and_return_all_conditional_losses"Ź
_tf_keras_layer°{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 6]}}

=regularization_losses
>	variables
?trainable_variables
@	keras_api
__call__
+&call_and_return_all_conditional_losses"ü
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

Alayers
Blayer_metrics
regularization_losses
	variables
trainable_variables
Clayer_regularization_losses
Dmetrics
Enon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object

F_inbound_nodes
G_outbound_nodes
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

L_inbound_nodes

!kernel
"bias
M_outbound_nodes
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
__call__
+&call_and_return_all_conditional_losses"Č
_tf_keras_layer®{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 120, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 400]}}

R_inbound_nodes

#kernel
$bias
S_outbound_nodes
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
 __call__
+”&call_and_return_all_conditional_losses"Ė
_tf_keras_layer±{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 120}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 120]}}

X_inbound_nodes

%kernel
&bias
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
¢__call__
+£&call_and_return_all_conditional_losses"Ģ
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

]layers
^layer_metrics
regularization_losses
	variables
trainable_variables
_layer_regularization_losses
`metrics
anon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

blayers
clayer_metrics
1regularization_losses
2	variables
3trainable_variables
dlayer_regularization_losses
emetrics
fnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

glayers
hlayer_metrics
5regularization_losses
6	variables
7trainable_variables
ilayer_regularization_losses
jmetrics
knon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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

llayers
mlayer_metrics
9regularization_losses
:	variables
;trainable_variables
nlayer_regularization_losses
ometrics
pnon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

qlayers
rlayer_metrics
=regularization_losses
>	variables
?trainable_variables
slayer_regularization_losses
tmetrics
unon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

vlayers
wlayer_metrics
Hregularization_losses
I	variables
Jtrainable_variables
xlayer_regularization_losses
ymetrics
znon_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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

{layers
|layer_metrics
Nregularization_losses
O	variables
Ptrainable_variables
}layer_regularization_losses
~metrics
non_trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
layers
layer_metrics
Tregularization_losses
U	variables
Vtrainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
 __call__
+”&call_and_return_all_conditional_losses
'”"call_and_return_conditional_losses"
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
layers
layer_metrics
Yregularization_losses
Z	variables
[trainable_variables
 layer_regularization_losses
metrics
non_trainable_variables
¢__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ü2Ł
(__inference_le_net5_layer_call_fn_278930
(__inference_le_net5_layer_call_fn_278905
(__inference_le_net5_layer_call_fn_278765
(__inference_le_net5_layer_call_fn_278790®
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
ē2ä
!__inference__wrapped_model_278189¾
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
Č2Å
C__inference_le_net5_layer_call_and_return_conditional_losses_278695
C__inference_le_net5_layer_call_and_return_conditional_losses_278740
C__inference_le_net5_layer_call_and_return_conditional_losses_278835
C__inference_le_net5_layer_call_and_return_conditional_losses_278880®
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
2
/__inference_zero_padding2d_layer_call_fn_278202ą
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
²2Æ
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_278196ą
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
×2Ō
2__inference_feature_extractor_layer_call_fn_278963
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
ņ2ļ
M__inference_feature_extractor_layer_call_and_return_conditional_losses_278950
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
ś2÷
+__inference_sequential_layer_call_fn_279051
+__inference_sequential_layer_call_fn_279122
+__inference_sequential_layer_call_fn_279139
+__inference_sequential_layer_call_fn_279034Ą
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
ę2ć
F__inference_sequential_layer_call_and_return_conditional_losses_279105
F__inference_sequential_layer_call_and_return_conditional_losses_279078
F__inference_sequential_layer_call_and_return_conditional_losses_278990
F__inference_sequential_layer_call_and_return_conditional_losses_279017Ą
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
3B1
$__inference_signature_wrapper_278650input_1
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
2
2__inference_average_pooling2d_layer_call_fn_278214ą
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
µ2²
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_278208ą
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
2
4__inference_average_pooling2d_1_layer_call_fn_278226ą
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
·2“
O__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_278220ą
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
Ņ2Ļ
(__inference_flatten_layer_call_fn_279150¢
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
ķ2ź
C__inference_flatten_layer_call_and_return_conditional_losses_279145¢
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
Š2Ķ
&__inference_dense_layer_call_fn_279170¢
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
ė2č
A__inference_dense_layer_call_and_return_conditional_losses_279161¢
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
Ņ2Ļ
(__inference_dense_1_layer_call_fn_279190¢
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
ķ2ź
C__inference_dense_1_layer_call_and_return_conditional_losses_279181¢
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
Ņ2Ļ
(__inference_dense_2_layer_call_fn_279210¢
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
ķ2ź
C__inference_dense_2_layer_call_and_return_conditional_losses_279201¢
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
  
!__inference__wrapped_model_278189{
 !"#$%&8¢5
.¢+
)&
input_1’’’’’’’’’
Ŗ "3Ŗ0
.
output_1"
output_1’’’’’’’’’
ņ
O__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_278220R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Ź
4__inference_average_pooling2d_1_layer_call_fn_278226R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’š
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_278208R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Č
2__inference_average_pooling2d_layer_call_fn_278214R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’£
C__inference_dense_1_layer_call_and_return_conditional_losses_279181\#$/¢,
%¢"
 
inputs’’’’’’’’’x
Ŗ "%¢"

0’’’’’’’’’T
 {
(__inference_dense_1_layer_call_fn_279190O#$/¢,
%¢"
 
inputs’’’’’’’’’x
Ŗ "’’’’’’’’’T£
C__inference_dense_2_layer_call_and_return_conditional_losses_279201\%&/¢,
%¢"
 
inputs’’’’’’’’’T
Ŗ "%¢"

0’’’’’’’’’

 {
(__inference_dense_2_layer_call_fn_279210O%&/¢,
%¢"
 
inputs’’’’’’’’’T
Ŗ "’’’’’’’’’
¢
A__inference_dense_layer_call_and_return_conditional_losses_279161]!"0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "%¢"

0’’’’’’’’’x
 z
&__inference_dense_layer_call_fn_279170P!"0¢-
&¢#
!
inputs’’’’’’’’’
Ŗ "’’’’’’’’’xŗ
M__inference_feature_extractor_layer_call_and_return_conditional_losses_278950i 2¢/
(¢%
# 
x’’’’’’’’’  
Ŗ "-¢*
# 
0’’’’’’’’’
 
2__inference_feature_extractor_layer_call_fn_278963\ 2¢/
(¢%
# 
x’’’’’’’’’  
Ŗ " ’’’’’’’’’Ø
C__inference_flatten_layer_call_and_return_conditional_losses_279145a7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "&¢#

0’’’’’’’’’
 
(__inference_flatten_layer_call_fn_279150T7¢4
-¢*
(%
inputs’’’’’’’’’
Ŗ "’’’’’’’’’ø
C__inference_le_net5_layer_call_and_return_conditional_losses_278695q
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’

 ø
C__inference_le_net5_layer_call_and_return_conditional_losses_278740q
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’

 ²
C__inference_le_net5_layer_call_and_return_conditional_losses_278835k
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p
Ŗ "%¢"

0’’’’’’’’’

 ²
C__inference_le_net5_layer_call_and_return_conditional_losses_278880k
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p 
Ŗ "%¢"

0’’’’’’’’’

 
(__inference_le_net5_layer_call_fn_278765d
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p
Ŗ "’’’’’’’’’

(__inference_le_net5_layer_call_fn_278790d
 !"#$%&<¢9
2¢/
)&
input_1’’’’’’’’’
p 
Ŗ "’’’’’’’’’

(__inference_le_net5_layer_call_fn_278905^
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p
Ŗ "’’’’’’’’’

(__inference_le_net5_layer_call_fn_278930^
 !"#$%&6¢3
,¢)
# 
x’’’’’’’’’
p 
Ŗ "’’’’’’’’’
Į
F__inference_sequential_layer_call_and_return_conditional_losses_278990w!"#$%&F¢C
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
 Į
F__inference_sequential_layer_call_and_return_conditional_losses_279017w!"#$%&F¢C
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
 ŗ
F__inference_sequential_layer_call_and_return_conditional_losses_279078p!"#$%&?¢<
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
 ŗ
F__inference_sequential_layer_call_and_return_conditional_losses_279105p!"#$%&?¢<
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
 
+__inference_sequential_layer_call_fn_279034j!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

+__inference_sequential_layer_call_fn_279051j!"#$%&F¢C
<¢9
/,
flatten_input’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’

+__inference_sequential_layer_call_fn_279122c!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p

 
Ŗ "’’’’’’’’’

+__inference_sequential_layer_call_fn_279139c!"#$%&?¢<
5¢2
(%
inputs’’’’’’’’’
p 

 
Ŗ "’’’’’’’’’
Æ
$__inference_signature_wrapper_278650
 !"#$%&C¢@
¢ 
9Ŗ6
4
input_1)&
input_1’’’’’’’’’"3Ŗ0
.
output_1"
output_1’’’’’’’’’
ķ
J__inference_zero_padding2d_layer_call_and_return_conditional_losses_278196R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ "H¢E
>;
04’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
 Å
/__inference_zero_padding2d_layer_call_fn_278202R¢O
H¢E
C@
inputs4’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’
Ŗ ";84’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’’