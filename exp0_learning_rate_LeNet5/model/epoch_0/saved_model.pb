з¬
њ£
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
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02unknown8ОЇ
≤
'le_net5/feature_extractor/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'le_net5/feature_extractor/conv2d/kernel
Ђ
;le_net5/feature_extractor/conv2d/kernel/Read/ReadVariableOpReadVariableOp'le_net5/feature_extractor/conv2d/kernel*&
_output_shapes
:*
dtype0
Ґ
%le_net5/feature_extractor/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%le_net5/feature_extractor/conv2d/bias
Ы
9le_net5/feature_extractor/conv2d/bias/Read/ReadVariableOpReadVariableOp%le_net5/feature_extractor/conv2d/bias*
_output_shapes
:*
dtype0
ґ
)le_net5/feature_extractor/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)le_net5/feature_extractor/conv2d_1/kernel
ѓ
=le_net5/feature_extractor/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp)le_net5/feature_extractor/conv2d_1/kernel*&
_output_shapes
:*
dtype0
¶
'le_net5/feature_extractor/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'le_net5/feature_extractor/conv2d_1/bias
Я
;le_net5/feature_extractor/conv2d_1/bias/Read/ReadVariableOpReadVariableOp'le_net5/feature_extractor/conv2d_1/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АМ*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
АМ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:М*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	МT*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	МT*
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
†'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*џ&
value—&Bќ& B«&
Ы
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

		keras_api
И
	
conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
	variables
trainable_variables
	keras_api
‘
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
	variables
trainable_variables
	keras_api
 
F
0
1
2
3
4
5
 6
!7
"8
#9
F
0
1
2
3
4
5
 6
!7
"8
#9
≠
$layer_metrics
regularization_losses
	variables
%layer_regularization_losses

&layers
'non_trainable_variables
trainable_variables
(metrics
 
 
h

kernel
bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
 

0
1
2
3

0
1
2
3
≠
9layer_metrics
regularization_losses
	variables
:layer_regularization_losses

;layers
<non_trainable_variables
trainable_variables
=metrics
{
>_inbound_nodes
?_outbound_nodes
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
С
D_inbound_nodes

kernel
bias
E_outbound_nodes
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
С
J_inbound_nodes

 kernel
!bias
K_outbound_nodes
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
|
P_inbound_nodes

"kernel
#bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
 
*
0
1
 2
!3
"4
#5
*
0
1
 2
!3
"4
#5
≠
Ulayer_metrics
regularization_losses
	variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
trainable_variables
Ymetrics
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
 
 

0
1
2
 
 
 

0
1

0
1
≠
Zlayer_metrics
)regularization_losses
*	variables
[layer_regularization_losses

\layers
]non_trainable_variables
+trainable_variables
^metrics
 
 
 
≠
_layer_metrics
-regularization_losses
.	variables
`layer_regularization_losses

alayers
bnon_trainable_variables
/trainable_variables
cmetrics
 

0
1

0
1
≠
dlayer_metrics
1regularization_losses
2	variables
elayer_regularization_losses

flayers
gnon_trainable_variables
3trainable_variables
hmetrics
 
 
 
≠
ilayer_metrics
5regularization_losses
6	variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
7trainable_variables
mmetrics
 
 


0
1
2
3
 
 
 
 
 
 
 
≠
nlayer_metrics
@regularization_losses
A	variables
olayer_regularization_losses

players
qnon_trainable_variables
Btrainable_variables
rmetrics
 
 
 

0
1

0
1
≠
slayer_metrics
Fregularization_losses
G	variables
tlayer_regularization_losses

ulayers
vnon_trainable_variables
Htrainable_variables
wmetrics
 
 
 

 0
!1

 0
!1
≠
xlayer_metrics
Lregularization_losses
M	variables
ylayer_regularization_losses

zlayers
{non_trainable_variables
Ntrainable_variables
|metrics
 
 

"0
#1

"0
#1
ѓ
}layer_metrics
Qregularization_losses
R	variables
~layer_regularization_losses

layers
Аnon_trainable_variables
Strainable_variables
Бmetrics
 
 

0
1
2
3
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
К
serving_default_input_1Placeholder*/
_output_shapes
:€€€€€€€€€*
dtype0*$
shape:€€€€€€€€€
њ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1'le_net5/feature_extractor/conv2d/kernel%le_net5/feature_extractor/conv2d/bias)le_net5/feature_extractor/conv2d_1/kernel'le_net5/feature_extractor/conv2d_1/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_62463
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ж
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
GPU 2J 8В *'
f"R 
__inference__traced_save_63068
Щ
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
GPU 2J 8В **
f%R#
!__inference__traced_restore_63108ым
Э
™
B__inference_dense_2_layer_call_and_return_conditional_losses_62125

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€T:::O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Х=
”
 __inference__wrapped_model_62018
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
identityИш
6le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp?le_net5_feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype028
6le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOpИ
'le_net5/feature_extractor/conv2d/Conv2DConv2Dinput_1>le_net5/feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2)
'le_net5/feature_extractor/conv2d/Conv2Dп
7le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp@le_net5_feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOpМ
(le_net5/feature_extractor/conv2d/BiasAddBiasAdd0le_net5/feature_extractor/conv2d/Conv2D:output:0?le_net5/feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(le_net5/feature_extractor/conv2d/BiasAdd√
%le_net5/feature_extractor/conv2d/TanhTanh1le_net5/feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%le_net5/feature_extractor/conv2d/TanhЦ
3le_net5/feature_extractor/average_pooling2d/AvgPoolAvgPool)le_net5/feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
25
3le_net5/feature_extractor/average_pooling2d/AvgPoolю
8le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAle_net5_feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02:
8le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOp√
)le_net5/feature_extractor/conv2d_1/Conv2DConv2D<le_net5/feature_extractor/average_pooling2d/AvgPool:output:0@le_net5/feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2+
)le_net5/feature_extractor/conv2d_1/Conv2Dх
9le_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBle_net5_feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02;
9le_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOpФ
*le_net5/feature_extractor/conv2d_1/BiasAddBiasAdd2le_net5/feature_extractor/conv2d_1/Conv2D:output:0Ale_net5/feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*le_net5/feature_extractor/conv2d_1/BiasAdd…
'le_net5/feature_extractor/conv2d_1/TanhTanh3le_net5/feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'le_net5/feature_extractor/conv2d_1/TanhЬ
5le_net5/feature_extractor/average_pooling2d_1/AvgPoolAvgPool+le_net5/feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
27
5le_net5/feature_extractor/average_pooling2d_1/AvgPoolХ
 le_net5/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2"
 le_net5/sequential/flatten/Constс
"le_net5/sequential/flatten/ReshapeReshape>le_net5/feature_extractor/average_pooling2d_1/AvgPool:output:0)le_net5/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2$
"le_net5/sequential/flatten/ReshapeЏ
.le_net5/sequential/dense/MatMul/ReadVariableOpReadVariableOp7le_net5_sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype020
.le_net5/sequential/dense/MatMul/ReadVariableOpд
le_net5/sequential/dense/MatMulMatMul+le_net5/sequential/flatten/Reshape:output:06le_net5/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2!
le_net5/sequential/dense/MatMulЎ
/le_net5/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp8le_net5_sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype021
/le_net5/sequential/dense/BiasAdd/ReadVariableOpж
 le_net5/sequential/dense/BiasAddBiasAdd)le_net5/sequential/dense/MatMul:product:07le_net5/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2"
 le_net5/sequential/dense/BiasAdd§
le_net5/sequential/dense/TanhTanh)le_net5/sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
le_net5/sequential/dense/Tanhя
0le_net5/sequential/dense_1/MatMul/ReadVariableOpReadVariableOp9le_net5_sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype022
0le_net5/sequential/dense_1/MatMul/ReadVariableOpя
!le_net5/sequential/dense_1/MatMulMatMul!le_net5/sequential/dense/Tanh:y:08le_net5/sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2#
!le_net5/sequential/dense_1/MatMulЁ
1le_net5/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp:le_net5_sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype023
1le_net5/sequential/dense_1/BiasAdd/ReadVariableOpн
"le_net5/sequential/dense_1/BiasAddBiasAdd+le_net5/sequential/dense_1/MatMul:product:09le_net5/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2$
"le_net5/sequential/dense_1/BiasAdd≤
"le_net5/sequential/dense_1/SoftmaxSoftmax+le_net5/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2$
"le_net5/sequential/dense_1/Softmaxё
0le_net5/sequential/dense_2/MatMul/ReadVariableOpReadVariableOp9le_net5_sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype022
0le_net5/sequential/dense_2/MatMul/ReadVariableOpк
!le_net5/sequential/dense_2/MatMulMatMul,le_net5/sequential/dense_1/Softmax:softmax:08le_net5/sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2#
!le_net5/sequential/dense_2/MatMulЁ
1le_net5/sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp:le_net5_sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype023
1le_net5/sequential/dense_2/BiasAdd/ReadVariableOpн
"le_net5/sequential/dense_2/BiasAddBiasAdd+le_net5/sequential/dense_2/MatMul:product:09le_net5/sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2$
"le_net5/sequential/dense_2/BiasAdd©
le_net5/sequential/dense_2/TanhTanh+le_net5/sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2!
le_net5/sequential/dense_2/Tanhw
IdentityIdentity#le_net5/sequential/dense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€:::::::::::X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
У
т
E__inference_sequential_layer_call_and_return_conditional_losses_62910
flatten_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstЗ
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2

dense/Tanh¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
dense_1/MatMul/ReadVariableOpУ
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/Softmax•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOpЮ
dense_2/MatMulMatMuldense_1/Softmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Tanhd
IdentityIdentitydense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€:::::::^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
ю
л
E__inference_sequential_layer_call_and_return_conditional_losses_62822

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstА
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2

dense/Tanh¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
dense_1/MatMul/ReadVariableOpУ
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/Softmax•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOpЮ
dense_2/MatMulMatMuldense_1/Softmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Tanhd
IdentityIdentitydense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€:::::::W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≤
™
B__inference_dense_1_layer_call_and_return_conditional_losses_62986

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€М:::P L
(
_output_shapes
:€€€€€€€€€М
 
_user_specified_nameinputs
Н
j
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_62036

inputs
identityґ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
AvgPoolЗ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
п
ї
*__inference_sequential_layer_call_fn_62839

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_621852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю
л
E__inference_sequential_layer_call_and_return_conditional_losses_62795

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstА
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2

dense/Tanh¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
dense_1/MatMul/ReadVariableOpУ
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/Softmax•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOpЮ
dense_2/MatMulMatMuldense_1/Softmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Tanhd
IdentityIdentitydense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€:::::::W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_62052

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ъ
Ш
L__inference_feature_extractor_layer_call_and_return_conditional_losses_62755
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityИ™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpі
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d/Tanh»
average_pooling2d/AvgPoolAvgPoolconv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpџ
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_1/Tanhќ
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPoolА
IdentityIdentity$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€:::::R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Д
¬
*__inference_sequential_layer_call_fn_62944
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_622222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
ъ
Ш
L__inference_feature_extractor_layer_call_and_return_conditional_losses_62261
x)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource
identityИ™
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpі
conv2d/Conv2DConv2Dx$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d/Conv2D°
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp§
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d/BiasAddu
conv2d/TanhTanhconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d/Tanh»
average_pooling2d/AvgPoolAvgPoolconv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool∞
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpџ
conv2d_1/Conv2DConv2D"average_pooling2d/AvgPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2
conv2d_1/Conv2DІ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpђ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_1/BiasAdd{
conv2d_1/TanhTanhconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
conv2d_1/Tanhќ
average_pooling2d_1/AvgPoolAvgPoolconv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPoolА
IdentityIdentity$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€:::::R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
ш
ґ
E__inference_sequential_layer_call_and_return_conditional_losses_62185

inputs
dense_62169
dense_62171
dense_1_62174
dense_1_62176
dense_2_62179
dense_2_62181
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCall—
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_620522
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_62169dense_62171*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€М*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_620712
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62174dense_1_62176*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_620982!
dense_1/StatefulPartitionedCallЃ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_62179dense_2_62181*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_621252!
dense_2/StatefulPartitionedCallа
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ6
Я
B__inference_le_net5_layer_call_and_return_conditional_losses_62642
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
identityИа
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOpк
feature_extractor/conv2d/Conv2DConv2Dx6feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D„
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpм
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 feature_extractor/conv2d/BiasAddЂ
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
feature_extractor/conv2d/Tanhю
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolж
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DЁ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpф
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2!
feature_extractor/conv2d_1/TanhД
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Const—
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/BiasAddМ
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/Tanh«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpњ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/Softmax∆
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp 
sequential/dense_2/MatMulMatMul$sequential/dense_1/Softmax:softmax:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/MatMul≈
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЌ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/BiasAddС
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/Tanho
IdentityIdentitysequential/dense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€:::::::::::R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
Џ
п
'__inference_le_net5_layer_call_fn_62710
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
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_le_net5_layer_call_and_return_conditional_losses_623882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
§
®
@__inference_dense_layer_call_and_return_conditional_losses_62966

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:€€€€€€€€€М2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
≥
O
3__inference_average_pooling2d_1_layer_call_fn_62042

inputs
identityп
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_620362
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я6
•
B__inference_le_net5_layer_call_and_return_conditional_losses_62506
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
identityИа
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOpр
feature_extractor/conv2d/Conv2DConv2Dinput_16feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D„
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpм
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 feature_extractor/conv2d/BiasAddЂ
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
feature_extractor/conv2d/Tanhю
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolж
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DЁ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpф
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2!
feature_extractor/conv2d_1/TanhД
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Const—
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/BiasAddМ
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/Tanh«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpњ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/Softmax∆
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp 
sequential/dense_2/MatMulMatMul$sequential/dense_1/Softmax:softmax:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/MatMul≈
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЌ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/BiasAddС
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/Tanho
IdentityIdentitysequential/dense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€:::::::::::X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Џ-
ц
!__inference__traced_restore_63108
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
identity_11ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ј
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*√
valueєBґB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names§
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesв
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

IdentityЈ
AssignVariableOpAssignVariableOp8assignvariableop_le_net5_feature_extractor_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1љ
AssignVariableOp_1AssignVariableOp8assignvariableop_1_le_net5_feature_extractor_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ѕ
AssignVariableOp_2AssignVariableOp<assignvariableop_2_le_net5_feature_extractor_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3њ
AssignVariableOp_3AssignVariableOp:assignvariableop_3_le_net5_feature_extractor_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¶
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7§
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¶
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9§
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10≠
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
ј"
Ы
__inference__traced_save_63068
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

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8d6d5859966f4c439caec818a6c717a8/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename±
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*√
valueєBґB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЮ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slicesћ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Bsavev2_le_net5_feature_extractor_conv2d_kernel_read_readvariableop@savev2_le_net5_feature_extractor_conv2d_bias_read_readvariableopDsavev2_le_net5_feature_extractor_conv2d_1_kernel_read_readvariableopBsavev2_le_net5_feature_extractor_conv2d_1_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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

identity_1Identity_1:output:0*{
_input_shapesj
h: :::::
АМ:М:	МT:T:T
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
::&"
 
_output_shapes
:
АМ:!

_output_shapes	
:М:%!

_output_shapes
:	МT: 
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
§
®
@__inference_dense_layer_call_and_return_conditional_losses_62071

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
MatMulН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpВ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2	
BiasAddY
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
Tanh]
IdentityIdentityTanh:y:0*
T0*(
_output_shapes
:€€€€€€€€€М2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А:::P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Д
¬
*__inference_sequential_layer_call_fn_62927
flatten_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall∞
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_621852
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
Ў
|
'__inference_dense_2_layer_call_fn_63015

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_621252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€T::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
ѓ
M
1__inference_average_pooling2d_layer_call_fn_62030

inputs
identityн
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_620242
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
м
х
'__inference_le_net5_layer_call_fn_62599
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
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_le_net5_layer_call_and_return_conditional_losses_623882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
ш
ґ
E__inference_sequential_layer_call_and_return_conditional_losses_62222

inputs
dense_62206
dense_62208
dense_1_62211
dense_1_62213
dense_2_62216
dense_2_62218
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCall—
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_620522
flatten/PartitionedCallЭ
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_62206dense_62208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€М*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_620712
dense/StatefulPartitionedCallђ
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_62211dense_1_62213*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_620982!
dense_1/StatefulPartitionedCallЃ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_62216dense_2_62218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_621252!
dense_2/StatefulPartitionedCallа
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
†
C
'__inference_flatten_layer_call_fn_62955

inputs
identityЅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_620522
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∆
с
#__inference_signature_wrapper_62463
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
identityИҐStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_620182
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ6
Я
B__inference_le_net5_layer_call_and_return_conditional_losses_62685
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
identityИа
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOpк
feature_extractor/conv2d/Conv2DConv2Dx6feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D„
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpм
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 feature_extractor/conv2d/BiasAddЂ
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
feature_extractor/conv2d/Tanhю
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolж
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DЁ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpф
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2!
feature_extractor/conv2d_1/TanhД
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Const—
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/BiasAddМ
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/Tanh«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpњ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/Softmax∆
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp 
sequential/dense_2/MatMulMatMul$sequential/dense_1/Softmax:softmax:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/MatMul≈
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЌ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/BiasAddС
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/Tanho
IdentityIdentitysequential/dense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€:::::::::::R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
≤
™
B__inference_dense_1_layer_call_and_return_conditional_losses_62098

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:T*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€М:::P L
(
_output_shapes
:€€€€€€€€€М
 
_user_specified_nameinputs
Л
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_62024

inputs
identityґ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
AvgPoolЗ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
п
ї
*__inference_sequential_layer_call_fn_62856

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_622222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Э
™
B__inference_dense_2_layer_call_and_return_conditional_losses_63006

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€T:::O K
'
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
Ї
^
B__inference_flatten_layer_call_and_return_conditional_losses_62950

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:W S
/
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ
п
'__inference_le_net5_layer_call_fn_62735
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
identityИҐStatefulPartitionedCall’
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_le_net5_layer_call_and_return_conditional_losses_623882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
У
т
E__inference_sequential_layer_call_and_return_conditional_losses_62883
flatten_input(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
flatten/ConstЗ
flatten/ReshapeReshapeflatten_inputflatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten/Reshape°
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02
dense/MatMul/ReadVariableOpШ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/MatMulЯ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
dense/BiasAdd/ReadVariableOpЪ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
dense/BiasAddk

dense/TanhTanhdense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2

dense/Tanh¶
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02
dense_1/MatMul/ReadVariableOpУ
dense_1/MatMulMatMuldense/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/MatMul§
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
dense_1/Softmax•
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02
dense_2/MatMul/ReadVariableOpЮ
dense_2/MatMulMatMuldense_1/Softmax:softmax:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/MatMul§
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02 
dense_2/BiasAdd/ReadVariableOp°
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/BiasAddp
dense_2/TanhTanhdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
dense_2/Tanhd
IdentityIdentitydense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:€€€€€€€€€:::::::^ Z
/
_output_shapes
:€€€€€€€€€
'
_user_specified_nameflatten_input
ь
•
B__inference_le_net5_layer_call_and_return_conditional_losses_62388
x
feature_extractor_62365
feature_extractor_62367
feature_extractor_62369
feature_extractor_62371
sequential_62374
sequential_62376
sequential_62378
sequential_62380
sequential_62382
sequential_62384
identityИҐ)feature_extractor/StatefulPartitionedCallҐ"sequential/StatefulPartitionedCallч
)feature_extractor/StatefulPartitionedCallStatefulPartitionedCallxfeature_extractor_62365feature_extractor_62367feature_extractor_62369feature_extractor_62371*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_feature_extractor_layer_call_and_return_conditional_losses_622612+
)feature_extractor/StatefulPartitionedCallЧ
"sequential/StatefulPartitionedCallStatefulPartitionedCall2feature_extractor/StatefulPartitionedCall:output:0sequential_62374sequential_62376sequential_62378sequential_62380sequential_62382sequential_62384*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_622222$
"sequential/StatefulPartitionedCall–
IdentityIdentity+sequential/StatefulPartitionedCall:output:0*^feature_extractor/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::2V
)feature_extractor/StatefulPartitionedCall)feature_extractor/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex
м
х
'__inference_le_net5_layer_call_fn_62574
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
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_le_net5_layer_call_and_return_conditional_losses_623882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ў
z
%__inference_dense_layer_call_fn_62975

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€М*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_620712
StatefulPartitionedCallП
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:€€€€€€€€€М2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Џ
|
'__inference_dense_1_layer_call_fn_62995

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€T*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_620982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€T2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€М::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€М
 
_user_specified_nameinputs
я6
•
B__inference_le_net5_layer_call_and_return_conditional_losses_62549
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
identityИа
.feature_extractor/conv2d/Conv2D/ReadVariableOpReadVariableOp7feature_extractor_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.feature_extractor/conv2d/Conv2D/ReadVariableOpр
feature_extractor/conv2d/Conv2DConv2Dinput_16feature_extractor/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2!
feature_extractor/conv2d/Conv2D„
/feature_extractor/conv2d/BiasAdd/ReadVariableOpReadVariableOp8feature_extractor_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/feature_extractor/conv2d/BiasAdd/ReadVariableOpм
 feature_extractor/conv2d/BiasAddBiasAdd(feature_extractor/conv2d/Conv2D:output:07feature_extractor/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2"
 feature_extractor/conv2d/BiasAddЂ
feature_extractor/conv2d/TanhTanh)feature_extractor/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2
feature_extractor/conv2d/Tanhю
+feature_extractor/average_pooling2d/AvgPoolAvgPool!feature_extractor/conv2d/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2-
+feature_extractor/average_pooling2d/AvgPoolж
0feature_extractor/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9feature_extractor_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0feature_extractor/conv2d_1/Conv2D/ReadVariableOp£
!feature_extractor/conv2d_1/Conv2DConv2D4feature_extractor/average_pooling2d/AvgPool:output:08feature_extractor/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€*
paddingVALID*
strides
2#
!feature_extractor/conv2d_1/Conv2DЁ
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:feature_extractor_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1feature_extractor/conv2d_1/BiasAdd/ReadVariableOpф
"feature_extractor/conv2d_1/BiasAddBiasAdd*feature_extractor/conv2d_1/Conv2D:output:09feature_extractor/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€2$
"feature_extractor/conv2d_1/BiasAdd±
feature_extractor/conv2d_1/TanhTanh+feature_extractor/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€2!
feature_extractor/conv2d_1/TanhД
-feature_extractor/average_pooling2d_1/AvgPoolAvgPool#feature_extractor/conv2d_1/Tanh:y:0*
T0*/
_output_shapes
:€€€€€€€€€*
ksize
*
paddingVALID*
strides
2/
-feature_extractor/average_pooling2d_1/AvgPoolЕ
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   2
sequential/flatten/Const—
sequential/flatten/ReshapeReshape6feature_extractor/average_pooling2d_1/AvgPool:output:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
sequential/flatten/Reshape¬
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
АМ*
dtype02(
&sequential/dense/MatMul/ReadVariableOpƒ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/MatMulј
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp∆
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/BiasAddМ
sequential/dense/TanhTanh!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€М2
sequential/dense/Tanh«
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	МT*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOpњ
sequential/dense_1/MatMulMatMulsequential/dense/Tanh:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/MatMul≈
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:T*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpЌ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/BiasAddЪ
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€T2
sequential/dense_1/Softmax∆
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:T
*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp 
sequential/dense_2/MatMulMatMul$sequential/dense_1/Softmax:softmax:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/MatMul≈
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOpЌ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/BiasAddС
sequential/dense_2/TanhTanh#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€
2
sequential/dense_2/Tanho
IdentityIdentitysequential/dense_2/Tanh:y:0*
T0*'
_output_shapes
:€€€€€€€€€
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:€€€€€€€€€:::::::::::X T
/
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Њ
Я
1__inference_feature_extractor_layer_call_fn_62768
x
unknown
	unknown_0
	unknown_1
	unknown_2
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *U
fPRN
L__inference_feature_extractor_layer_call_and_return_conditional_losses_622612
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:€€€€€€€€€

_user_specified_namex"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*≥
serving_defaultЯ
C
input_18
serving_default_input_1:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€
tensorflow/serving/predict:™Б
н
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
В__call__
Г_default_save_signature
+Д&call_and_return_all_conditional_losses"х
_tf_keras_modelџ{"class_name": "LeNet5", "name": "le_net5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "LeNet5"}}
И
		keras_api"ц
_tf_keras_layer№{"class_name": "ZeroPadding2D", "name": "zero_padding2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
л
	
conv1

conv1_pool
	conv2

conv2_pool
regularization_losses
	variables
trainable_variables
	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"§
_tf_keras_layerК{"class_name": "FeatureExtractor", "name": "feature_extractor", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
Я 
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
regularization_losses
	variables
trainable_variables
	keras_api
З__call__
+И&call_and_return_all_conditional_losses"М
_tf_keras_sequentialн{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 4, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 140, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4, 4, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 140, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
 "
trackable_list_wrapper
f
0
1
2
3
4
5
 6
!7
"8
#9"
trackable_list_wrapper
f
0
1
2
3
4
5
 6
!7
"8
#9"
trackable_list_wrapper
ќ
$layer_metrics
regularization_losses
	variables
%layer_regularization_losses

&layers
'non_trainable_variables
trainable_variables
(metrics
В__call__
Г_default_save_signature
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
-
Йserving_default"
signature_map
"
_generic_user_object
н	

kernel
bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
К__call__
+Л&call_and_return_all_conditional_losses"∆
_tf_keras_layerђ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 6, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 28, 28, 1]}}
Й
-regularization_losses
.	variables
/trainable_variables
0	keras_api
М__call__
+Н&call_and_return_all_conditional_losses"ш
_tf_keras_layerё{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
т	

kernel
bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
О__call__
+П&call_and_return_all_conditional_losses"Ћ
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 12, 12, 6]}}
Н
5regularization_losses
6	variables
7trainable_variables
8	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"ь
_tf_keras_layerв{"class_name": "AveragePooling2D", "name": "average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
∞
9layer_metrics
regularization_losses
	variables
:layer_regularization_losses

;layers
<non_trainable_variables
trainable_variables
=metrics
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
Н
>_inbound_nodes
?_outbound_nodes
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"”
_tf_keras_layerє{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Щ
D_inbound_nodes

kernel
bias
E_outbound_nodes
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 140, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 256]}}
Я
J_inbound_nodes

 kernel
!bias
K_outbound_nodes
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"ѕ
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 84, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 140}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 140]}}
Е
P_inbound_nodes

"kernel
#bias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses" 
_tf_keras_layer∞{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 84}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 84]}}
 "
trackable_list_wrapper
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
J
0
1
 2
!3
"4
#5"
trackable_list_wrapper
∞
Ulayer_metrics
regularization_losses
	variables
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
trainable_variables
Ymetrics
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
A:?2'le_net5/feature_extractor/conv2d/kernel
3:12%le_net5/feature_extractor/conv2d/bias
C:A2)le_net5/feature_extractor/conv2d_1/kernel
5:32'le_net5/feature_extractor/conv2d_1/bias
 :
АМ2dense/kernel
:М2
dense/bias
!:	МT2dense_1/kernel
:T2dense_1/bias
 :T
2dense_2/kernel
:
2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
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
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
Zlayer_metrics
)regularization_losses
*	variables
[layer_regularization_losses

\layers
]non_trainable_variables
+trainable_variables
^metrics
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
_layer_metrics
-regularization_losses
.	variables
`layer_regularization_losses

alayers
bnon_trainable_variables
/trainable_variables
cmetrics
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
dlayer_metrics
1regularization_losses
2	variables
elayer_regularization_losses

flayers
gnon_trainable_variables
3trainable_variables
hmetrics
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
ilayer_metrics
5regularization_losses
6	variables
jlayer_regularization_losses

klayers
lnon_trainable_variables
7trainable_variables
mmetrics
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<

0
1
2
3"
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
∞
nlayer_metrics
@regularization_losses
A	variables
olayer_regularization_losses

players
qnon_trainable_variables
Btrainable_variables
rmetrics
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
slayer_metrics
Fregularization_losses
G	variables
tlayer_regularization_losses

ulayers
vnon_trainable_variables
Htrainable_variables
wmetrics
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
∞
xlayer_metrics
Lregularization_losses
M	variables
ylayer_regularization_losses

zlayers
{non_trainable_variables
Ntrainable_variables
|metrics
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
≤
}layer_metrics
Qregularization_losses
R	variables
~layer_regularization_losses

layers
Аnon_trainable_variables
Strainable_variables
Бmetrics
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
Ў2’
'__inference_le_net5_layer_call_fn_62599
'__inference_le_net5_layer_call_fn_62574
'__inference_le_net5_layer_call_fn_62710
'__inference_le_net5_layer_call_fn_62735Ѓ
•≤°
FullArgSpec$
argsЪ
jself
jx

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2г
 __inference__wrapped_model_62018Њ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *.Ґ+
)К&
input_1€€€€€€€€€
ƒ2Ѕ
B__inference_le_net5_layer_call_and_return_conditional_losses_62685
B__inference_le_net5_layer_call_and_return_conditional_losses_62549
B__inference_le_net5_layer_call_and_return_conditional_losses_62642
B__inference_le_net5_layer_call_and_return_conditional_losses_62506Ѓ
•≤°
FullArgSpec$
argsЪ
jself
jx

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
1__inference_feature_extractor_layer_call_fn_62768Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
с2о
L__inference_feature_extractor_layer_call_and_return_conditional_losses_62755Э
Ф≤Р
FullArgSpec
argsЪ
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ц2у
*__inference_sequential_layer_call_fn_62856
*__inference_sequential_layer_call_fn_62839
*__inference_sequential_layer_call_fn_62927
*__inference_sequential_layer_call_fn_62944ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
E__inference_sequential_layer_call_and_return_conditional_losses_62883
E__inference_sequential_layer_call_and_return_conditional_losses_62910
E__inference_sequential_layer_call_and_return_conditional_losses_62822
E__inference_sequential_layer_call_and_return_conditional_losses_62795ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
2B0
#__inference_signature_wrapper_62463input_1
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Щ2Ц
1__inference_average_pooling2d_layer_call_fn_62030а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
і2±
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_62024а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ы2Ш
3__inference_average_pooling2d_1_layer_call_fn_62042а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
ґ2≥
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_62036а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
—2ќ
'__inference_flatten_layer_call_fn_62955Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_flatten_layer_call_and_return_conditional_losses_62950Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_dense_layer_call_fn_62975Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_dense_layer_call_and_return_conditional_losses_62966Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_1_layer_call_fn_62995Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_1_layer_call_and_return_conditional_losses_62986Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_dense_2_layer_call_fn_63015Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_dense_2_layer_call_and_return_conditional_losses_63006Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 Я
 __inference__wrapped_model_62018{
 !"#8Ґ5
.Ґ+
)К&
input_1€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€
с
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_62036ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ …
3__inference_average_pooling2d_1_layer_call_fn_62042СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€п
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_62024ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ «
1__inference_average_pooling2d_layer_call_fn_62030СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€£
B__inference_dense_1_layer_call_and_return_conditional_losses_62986] !0Ґ-
&Ґ#
!К
inputs€€€€€€€€€М
™ "%Ґ"
К
0€€€€€€€€€T
Ъ {
'__inference_dense_1_layer_call_fn_62995P !0Ґ-
&Ґ#
!К
inputs€€€€€€€€€М
™ "К€€€€€€€€€TҐ
B__inference_dense_2_layer_call_and_return_conditional_losses_63006\"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€T
™ "%Ґ"
К
0€€€€€€€€€

Ъ z
'__inference_dense_2_layer_call_fn_63015O"#/Ґ,
%Ґ"
 К
inputs€€€€€€€€€T
™ "К€€€€€€€€€
Ґ
@__inference_dense_layer_call_and_return_conditional_losses_62966^0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "&Ґ#
К
0€€€€€€€€€М
Ъ z
%__inference_dense_layer_call_fn_62975Q0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€Мє
L__inference_feature_extractor_layer_call_and_return_conditional_losses_62755i2Ґ/
(Ґ%
#К 
x€€€€€€€€€
™ "-Ґ*
#К 
0€€€€€€€€€
Ъ С
1__inference_feature_extractor_layer_call_fn_62768\2Ґ/
(Ґ%
#К 
x€€€€€€€€€
™ " К€€€€€€€€€І
B__inference_flatten_layer_call_and_return_conditional_losses_62950a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "&Ґ#
К
0€€€€€€€€€А
Ъ 
'__inference_flatten_layer_call_fn_62955T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€
™ "К€€€€€€€€€АЈ
B__inference_le_net5_layer_call_and_return_conditional_losses_62506q
 !"#<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€

Ъ Ј
B__inference_le_net5_layer_call_and_return_conditional_losses_62549q
 !"#<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€

Ъ ±
B__inference_le_net5_layer_call_and_return_conditional_losses_62642k
 !"#6Ґ3
,Ґ)
#К 
x€€€€€€€€€
p
™ "%Ґ"
К
0€€€€€€€€€

Ъ ±
B__inference_le_net5_layer_call_and_return_conditional_losses_62685k
 !"#6Ґ3
,Ґ)
#К 
x€€€€€€€€€
p 
™ "%Ґ"
К
0€€€€€€€€€

Ъ П
'__inference_le_net5_layer_call_fn_62574d
 !"#<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p
™ "К€€€€€€€€€
П
'__inference_le_net5_layer_call_fn_62599d
 !"#<Ґ9
2Ґ/
)К&
input_1€€€€€€€€€
p 
™ "К€€€€€€€€€
Й
'__inference_le_net5_layer_call_fn_62710^
 !"#6Ґ3
,Ґ)
#К 
x€€€€€€€€€
p
™ "К€€€€€€€€€
Й
'__inference_le_net5_layer_call_fn_62735^
 !"#6Ґ3
,Ґ)
#К 
x€€€€€€€€€
p 
™ "К€€€€€€€€€
є
E__inference_sequential_layer_call_and_return_conditional_losses_62795p !"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ є
E__inference_sequential_layer_call_and_return_conditional_losses_62822p !"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ ј
E__inference_sequential_layer_call_and_return_conditional_losses_62883w !"#FҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ ј
E__inference_sequential_layer_call_and_return_conditional_losses_62910w !"#FҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p 

 
™ "%Ґ"
К
0€€€€€€€€€

Ъ С
*__inference_sequential_layer_call_fn_62839c !"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p

 
™ "К€€€€€€€€€
С
*__inference_sequential_layer_call_fn_62856c !"#?Ґ<
5Ґ2
(К%
inputs€€€€€€€€€
p 

 
™ "К€€€€€€€€€
Ш
*__inference_sequential_layer_call_fn_62927j !"#FҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p

 
™ "К€€€€€€€€€
Ш
*__inference_sequential_layer_call_fn_62944j !"#FҐC
<Ґ9
/К,
flatten_input€€€€€€€€€
p 

 
™ "К€€€€€€€€€
Ѓ
#__inference_signature_wrapper_62463Ж
 !"#CҐ@
Ґ 
9™6
4
input_1)К&
input_1€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€
