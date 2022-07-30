### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ f21a6cda-9a7e-471c-b80e-f7afe3931e5d
using LinearAlgebra

# ╔═╡ be266b88-0ee8-11ed-17f8-7b18be34e038
md"""
# Packages
"""

# ╔═╡ ebcbe195-638f-439d-8fa6-24df225134e2
md"""
# Utils
"""

# ╔═╡ 0810d2b4-8d69-4588-b063-24f5798bcf10
verify_magnitude_sum(zs::Number...) = sum((z->abs(z)^2).(zs)) ≈ 1

# ╔═╡ d1f14ea8-2ba6-40c6-a560-4efb08b647df
struct Qubit
	# |Ψ⟩ = α|0⟩ + β|1⟩
	# |α|² + |β|² = 1
	α::Complex
	β::Complex
	Qubit(α, β) = verify_magnitude_sum(α, β) ? new(α, β) : error("Invalid Probability Amplitudes")
	Qubit(v::Matrix{ComplexF64}) = verify_magnitude_sum(v[1:2]...) ? new(v[1:2]...) : error("Invalid Probability Amplitudes")
end

# ╔═╡ cc136271-39bf-4a0a-a57d-56c9168c726f
qubit_vector(q::Qubit)::Matrix = float.(reshape([q.α, q.β], (2, 1)))

# ╔═╡ 8a0e755f-205d-488f-b58e-b340c617ff98
multi_qubit_vector(qs::Qubit...)::Matrix = kron(qubit_vector.(qs)...)

# ╔═╡ 40551fe4-3b70-46ea-847b-e0e0feef477c
struct DoubleQubit
	# |Ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩
	# |α|² + |β|² + |γ|² + |δ|² = 1
	α::Complex
	β::Complex
	γ::Complex
	δ::Complex
	DoubleQubit(α, β, γ, δ) = verify_magnitude_sum(α, β, γ, δ) ? new(α, β, γ, δ) : error("Invalid Probability Amplitudes")
	DoubleQubit(v::Matrix{ComplexF64}) = verify_magnitude_sum(v[1:4]...) ? new(v[1:4]...) : error("Invalid Probability Amplitudes")
end

# ╔═╡ c6e5e683-5c8f-418f-ba04-f2dbdb06134f
struct TripleQubit
	# |Ψ⟩ = α|000⟩ + β|001⟩ + γ|010⟩ + δ|011⟩ + ϵ|100⟩ + ζ|101⟩ + η|110⟩ + θ|111⟩
	# |α|² + |β|² + |γ|² + |δ|² + |ϵ|² + |ζ|² + |η|² + |θ|² = 1
	α::Complex
	β::Complex
	γ::Complex
	δ::Complex
	ϵ::Complex
	ζ::Complex
	η::Complex
	θ::Complex
	TripleQubit(α, β, γ, δ, ϵ, ζ, η, θ) = verify_magnitude_sum(α, β, γ, δ, ϵ, ζ, η, θ) ? new(α, β, γ, δ, ϵ, ζ, η, θ) : error("Invalid Probability Amplitudes")
	TripleQubit(v::Matrix{ComplexF64}) = verify_magnitude_sum(v[1:8]...) ? new(v[1:8]...) : error("Invalid Probability Amplitudes")
end

# ╔═╡ 9cecbf6c-d6a5-4ec2-8647-56da7835c3f3
struct NQubit{N}
	# |ψ⟩ = ∑ᵢαᵢ|bin(i)⟩
	# ∑ᵢ|αᵢ|² = 1
	coefficients::Matrix{ComplexF64}
	NQubit(qubits::Qubit...) = length(qubits)>1 ? new{length(qubits)}(multi_qubit_vector(qubits...)) : new{length(qubits)}(qubit_vector(qubits[1]))
end

# ╔═╡ c136d6c1-67c5-42d3-a039-895ebb6caf4e
begin
	import Base.*
	a::Matrix{ComplexF64} * b::Qubit = Qubit(a * qubit_vector(b))
	a::Matrix{ComplexF64} * b::NQubit = a * b.coefficients
end

# ╔═╡ 785460c9-e799-416f-8a34-8a0f7dabe1b5
begin
	custom_round(z, n_digits=10) = round(real(z), digits=n_digits) + round(imag(z), digits=n_digits)
	R_x(θ::ComplexF64) = [cos(θ/2) -im*sin(θ/2); -im*sin(θ/2) cos(θ/2)]
	R_y(θ::ComplexF64) = [cos(θ/2) -sin(θ/2); sin(θ/2) cos(θ/2)]
	R_z(θ::ComplexF64) = [exp(-im*θ/2) 0; 0 exp(im*θ/2)]
	compose(α::Real, θ₀::Real, θ₁::Real, θ₂::Real) = custom_round.(exp(im*α)*R_z(ComplexF64(θ₀))*R_y(ComplexF64(θ₁))*R_z(ComplexF64(θ₂)))
	decompose(U::Matrix{ComplexF64}) = begin
		α = atan(imag(det(U)),real(det(U)))/2
		V = exp(-im*α)*U
		θ₁ = abs(V[1, 1])≥abs(V[1, 2]) ? 2*acos(abs(V[1, 1])) : 2*asin(abs(V[1, 2]))
		if cos(θ₁/2) == 0
			θ₀ = atan(imag(V[2, 1]/sin(θ₁/2)), real(mag(V[2, 1]/sin(θ₁/2))))
			θ₀ = -θ₂
		elseif sin(θ₁/2) == 0
			θ₀ = atan(imag(V[2, 2]/cos(θ₁/2)), real(V[2, 2]/cos(θ₁/2)))
			θ₂ = θ₀
		else
			θ₀ = atan(imag(V[2, 2]/cos(θ₁/2)), real(V[2, 2]/cos(θ₁/2)))+atan(imag(V[2, 1]/sin(θ₁/2)), real(V[2, 1]/sin(θ₁/2)))
			θ₂ = 2*atan(imag(V[2, 2]/cos(θ₁/2)), real(V[2, 2]/cos(θ₁/2)))-θ₀
		end
		return custom_round.((α, θ₀, θ₁, θ₂))
	end
end

# ╔═╡ 9f48d5d9-320c-4a82-9f88-960668887f55
begin
	linear_superposition_representation(ψ::Matrix) = join(["($(ψ[i]))|$(lpad(string(i-1, base=2), Int(log2(length(ψ))), "0"))⟩" for i in 1:length(ψ)], "+")
	linear_superposition_representation(ψ::NQubit) = join(["($(ψ.coefficients[i]))|$(lpad(string(i-1, base=2), Int(log2(length(ψ.coefficients))), "0"))⟩" for i in 1:length(ψ.coefficients)], "+")
end

# ╔═╡ 9fdab464-aff2-44ba-b058-ac6fa56293a9
begin
	probabilities(ψ::Matrix) = vcat(["|$(lpad(string(i-1, base=2), Int(log2(length(ψ))), "0"))⟩: $(round(abs(ψ[i])^2, digits=5))" for i in 1:length(ψ)]...)
	probabilities(ψ::NQubit) = vcat(["|$(lpad(string(i-1, base=2), Int(log2(length(ψ.coefficients))), "0"))⟩: $(round(abs(ψ.coefficients[i])^2, digits=5))" for i in 1:length(ψ.coefficients)]...)
end

# ╔═╡ c0a94f4f-d3de-486f-a882-28627cbfce1e
md"""
# Gates
"""

# ╔═╡ f065fd22-20e4-48e8-b2b4-b88b7f0fae18
md"""
## Unitary Gates
"""

# ╔═╡ 0404d066-999c-4b1e-9434-010c487a0968
I::Matrix{ComplexF64} = [1 0; 0 1]

# ╔═╡ 5a52f5e3-b9aa-485a-9689-2d5d58b5cfe0
H::Matrix{ComplexF64} = [1/√(2) 1/√(2); 1/√(2) -1/√(2)]

# ╔═╡ 7b5461bd-71bb-4155-b886-1285bc842078
X::Matrix{ComplexF64} = [0 1; 1 0]

# ╔═╡ da1cba03-b50d-4f37-b94c-5f6538062041
Y::Matrix{ComplexF64} = [0 -im; im 0]

# ╔═╡ 7f768046-8e1b-4429-9727-958ef902f86e
Z::Matrix{ComplexF64} = [1 0; 0 -1]

# ╔═╡ bf204d31-c0ff-4313-abb7-4f65aacb1d1b
S::Matrix{ComplexF64} = [1 0; 0 im]

# ╔═╡ 3ce5ac59-22e0-4477-a5e0-4db2c97e12b3
T::Matrix{ComplexF64} = [1 0; 0 sqrt(im)]

# ╔═╡ 7589200e-711a-4c03-95d5-cdc422e82289
md"""
## Binary Gates
"""

# ╔═╡ 3157d8e0-fb97-4774-b80c-3a455d9d558f
CU(U::Matrix{ComplexF64}) = hvcat(
	(2, 2), 
	[1 0; 0 1], 
	[0 0; 0 0], 
	[0 0; 0 0], 
	U
)

# ╔═╡ 66c08c17-f32e-44c6-8d60-63dc25172812
CX::Matrix{ComplexF64} = CU(X)

# ╔═╡ 74e48ac3-8743-4f30-a5c2-c127007902ad
CZ::Matrix{ComplexF64} = CU(Z)

# ╔═╡ cccefe0c-95ca-40b3-8642-d3edd1a14081
CS::Matrix{ComplexF64} = CU(S)

# ╔═╡ 0761a159-7268-46ca-a9c7-b58efb2db7dc
CH::Matrix{ComplexF64} = CU(H)

# ╔═╡ 1a324973-1f51-4111-a776-8265f9ce092d
SWAP::Matrix{ComplexF64} = [1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

# ╔═╡ 8996891f-087a-44cc-87ed-d2f94ba09187
md"""
## Ternary Gates
"""

# ╔═╡ 356e92c0-66e9-49e1-9938-462c63b2f4bd
CCU(U::Matrix{ComplexF64}) = hvcat(
	(2, 2),
	[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
	[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
	[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
	CU(U)
)

# ╔═╡ 3ce4fd45-3e37-4263-84ec-7ed04828edb4
CB(B::Matrix{ComplexF64}) = hvcat(
	(2, 2),
	[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1],
	[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
	[0 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0],
	B
)

# ╔═╡ 9db796f5-701e-42b5-81ea-78bae2c30132
CCX::Matrix{ComplexF64} = CCU(X)

# ╔═╡ a9c85502-80d8-4363-84ad-8cd8e3486c15
CSWAP::Matrix{ComplexF64} = CB(SWAP)

# ╔═╡ 21b61975-72c5-4777-87e0-65ce4a365e7a
md"""
# Levels
"""

# ╔═╡ 5bc686c8-2d32-4e25-985e-ca95f4ba5da6
md"""
## World 1 - Basic Gates
"""

# ╔═╡ 3d8715d3-8421-4c73-9f93-a4703c3745fa
md"""
### Level 1-1: Introduction

#### Instructions
 - Place the Qubit on the circuit.
 - The Qubit has been prepared to collapse to $|0⟩$ with probability $\frac{1}{2}$ and $|1⟩$ with probability $\frac{1}{2}$
 - Goal:
   -  $P(|0⟩) = \frac{1}{2}$
   -  $P(|1⟩) = \frac{1}{2}$

#### Gates
 - None

#### Qubits
 -  $|ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $|ψ⟩ = |ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$

#### Solution
 -  $|ψ^′⟩ = |ψ⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}$
 - Measured states:
   -  $P(|0⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$
   -  $P(|1⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$

"""

# ╔═╡ 3de435e2-0463-4859-a083-772e96e2a3f7
NQubit(
	Qubit(
		1/sqrt(2), 
		1/sqrt(2)
	)
) |> probabilities

# ╔═╡ 57730711-ba0a-48a0-9c15-b627c6add2cb
md"""
### Level 1-2: Pauli-X gates

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|0⟩) = 0$
   -  $P(|1⟩) = 1$

#### Gates
 -  $X$: $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = 1|0⟩+0|1⟩$
 -  $|ψ⟩ = |ψ₁⟩ = 1|0⟩+0|1⟩$

#### Solution
 -  $|ψ^′⟩ = X|ψ⟩ = 0|0⟩+1|1⟩$
 -  $X\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$
 - Measured states:
   -  $P(|0⟩) = |0|² = 0$
   -  $P(|1⟩) = |1|² = 1$

"""

# ╔═╡ cfdc29e6-2aa6-4c54-8d43-05856a0aba7b
X * NQubit(
	Qubit(
		1, 
		0
	)
) |> probabilities

# ╔═╡ 5c9a6968-d36f-4219-90bf-cc0753b38457
md"""
### Level 1-3: Controlled Not (CX) gates

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|00⟩) = 0$
   -  $P(|01⟩) = 0$
   -  $P(|10⟩) = 1$
   -  $P(|11⟩) = 0$

#### Gates
 -  $CX$: $\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = 0|0⟩+1|1⟩$
 -  $|ψ₂⟩ = 0|0⟩+1|1⟩$
 -  $|ψ⟩ = |ψ₁⟩⊗|ψ₂⟩ = 0|00⟩+0|01⟩+0|10⟩+1|11⟩$

#### Solution
 -  $|ψ^′⟩ = CX|ψ⟩ = 0|00⟩+0|01⟩+1|10⟩+0|11⟩$
 -  $CX\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}$
 - Measured states:
   -  $P(|00⟩) = |0|² = 0$
   -  $P(|01⟩) = |0|² = 0$
   -  $P(|10⟩) = |1|² = 1$
   -  $P(|11⟩) = |0|² = 0$

"""

# ╔═╡ 2a4310c1-4250-4c9c-a7e7-7f9d894c21d3
CX * NQubit(
	Qubit(
		0, 
		1
	),
	Qubit(
		0,
		1
	)
) |> probabilities

# ╔═╡ b05eb619-9a20-42f0-bec8-6f3b0f37c34c
md"""
### Level 1-4: Hadamard (H) gates

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|0⟩) = \frac{1}{2}$
   -  $P(|1⟩) = \frac{1}{2}$

#### Gates
 -  $H$: $\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = 1|0⟩+0|1⟩$
 -  $|ψ⟩ = |ψ₁⟩ = 1|0⟩+0|1⟩$

#### Solution
 -  $|ψ^′⟩ = H|ψ⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $H\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}$
 - Measured states:
   -  $P(|0⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$
   -  $P(|1⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$

"""

# ╔═╡ 4de05494-50e9-480c-96d5-5f5292c09b8d
H * NQubit(
	Qubit(
		1, 
		0
	)
) |> probabilities

# ╔═╡ e36db4ac-d02a-49aa-8966-90428ddfa6b6
md"""
### Level 1-5: Hadamard (H) gates II

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|0⟩) = 1$
   -  $P(|1⟩) = 0$

#### Gates
 -  $H$: $\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $|ψ⟩ = |ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$

#### Solution
 -  $|ψ^′⟩ = H|ψ⟩ = 1|0⟩+0|1⟩$
 -  $H\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
 - Measured states:
   -  $P(|0⟩) = |1|² = 1$
   -  $P(|1⟩) = |0|² = 0$

"""

# ╔═╡ eac3549b-b79b-4255-8747-ad202feaf9c9
H * NQubit(
	Qubit(
		1/sqrt(2), 
		1/sqrt(2)
	)
) |> probabilities

# ╔═╡ 2d9180cd-79d0-4398-94b4-8e28c21ca24d
md"""
### Level 1-6: Controlled Hadamard (CH) gates

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|00⟩) = \frac{1}{2}$
   -  $P(|01⟩) = 0$
   -  $P(|10⟩) = \frac{1}{4}$
   -  $P(|11⟩) = \frac{1}{4}$

#### Gates
 -  $CH$: $\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $|ψ₂⟩ = 1|0⟩+0|1⟩$
 -  $|ψ⟩ = |ψ₁⟩⊗|ψ₂⟩ = \frac{1}{\sqrt{2}}|00⟩+0|01⟩+\frac{1}{\sqrt{2}}|10⟩+0|11⟩$

#### Solution
 -  $|ψ^′⟩ = CH|ψ⟩ = \frac{1}{\sqrt{2}}|00⟩+0|01⟩+\frac{1}{2}|10⟩+\frac{1}{2}|11⟩$
 -  $CH\begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{\sqrt{2}} \\ 0 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{2} \\ \frac{1}{2} \end{bmatrix}$
 - Measured states:
   -  $P(|00⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$
   -  $P(|01⟩) = |0|² = 0$
   -  $P(|10⟩) = |\frac{1}{2}|² = \frac{1}{4}$
   -  $P(|11⟩) = |\frac{1}{2}|² = \frac{1}{4}$

"""

# ╔═╡ 525685e9-a15b-492a-bc18-4f81ad0414d9
CH * NQubit(
	Qubit(
		1/sqrt(2), 
		1/sqrt(2)
	),
	Qubit(
		1,
		0
	)
) |> probabilities

# ╔═╡ 27a89157-745c-4e4d-958a-5886e812d284
md"""
### Level 1-7: Entanglement

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|00⟩) = \frac{1}{2}$
   -  $P(|01⟩) = 0$
   -  $P(|10⟩) = 0$
   -  $P(|11⟩) = \frac{1}{2}$

#### Gates
 -  $H$: $\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}$
 -  $CX$: $\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = 1|0⟩+0|1⟩$
 -  $|ψ₂⟩ = 1|0⟩+0|1⟩$
 -  $|ψ⟩ = |ψ₁⟩⊗|ψ₂⟩ = 1|00⟩+0|01⟩+0|10⟩+0|11⟩$

#### Solution
 -  $|ψ^′⟩ = CX(H|ψ₁⟩⊗|ψ₂⟩) = \frac{1}{\sqrt{2}}|00⟩+0|01⟩+0|10⟩+\frac{1}{\sqrt{2}}|11⟩$
 -  $CX(H\begin{bmatrix} 1 \\ 0 \end{bmatrix} ⊗ \begin{bmatrix} 1 \\ 0 \end{bmatrix}) = CX(\begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} ⊗ \begin{bmatrix} 1 \\ 0 \end{bmatrix}) = CX(\begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix} ⊗ \begin{bmatrix} 1 \\ 0 \end{bmatrix}) = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ \frac{1}{\sqrt{2}} \\ 0\end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ 0 \\ 0 \\ \frac{1}{\sqrt{2}}\end{bmatrix}$
 - Measured states:
   -  $P(|00⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$
   -  $P(|01⟩) = |0|² = 0$
   -  $P(|10⟩) = |0|² = 0$
   -  $P(|11⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$

"""

# ╔═╡ a59ef905-8c2b-480a-888f-87d8aad3ee03
CX * NQubit(
	H * Qubit(
		1, 
		0
	),
	Qubit(
		1,
		0
	)
) |> probabilities

# ╔═╡ b2ceb23e-8eaa-4fd3-87fe-94d71978cfc0
md"""
### Level 1-8: Multiple Controlled Gates

#### Instructions
 - Place the Qubit on the circuit.
 - Goal:
   -  $P(|00⟩) = \frac{1}{4}$
   -  $P(|01⟩) = \frac{1}{4}$
   -  $P(|10⟩) = 0$
   -  $P(|11⟩) = \frac{1}{2}$

#### Gates
 -  $CH$: $\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}$
 -  $CX$: $\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$

#### Qubits
 -  $|ψ₁⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $|ψ₂⟩ = \frac{1}{\sqrt{2}}|0⟩+\frac{1}{\sqrt{2}}|1⟩$
 -  $|ψ⟩ = |ψ₁⟩⊗|ψ₂⟩ = \frac{1}{2}|00⟩+\frac{1}{2}|01⟩+\frac{1}{2}|10⟩+\frac{1}{2}|11⟩$

#### Solution
 -  $|ψ^′⟩ = CH|ψ⟩ = \frac{1}{2}|00⟩+\frac{1}{2}|01⟩+0|10⟩+\frac{1}{\sqrt{2}}|11⟩$
 -  $CXCH\begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ 0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\ 0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}\begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \\ \frac{1}{2} \end{bmatrix} = \begin{bmatrix} \frac{1}{2} \\ \frac{1}{2} \\ 0 \\ \frac{1}{\sqrt{2}} \end{bmatrix}$
 - Measured states:
   -  $P(|00⟩) = |\frac{1}{2}|² = \frac{1}{4}$
   -  $P(|01⟩) = |\frac{1}{2}|² = \frac{1}{4}$
   -  $P(|10⟩) = |0|² = 0$
   -  $P(|11⟩) = |\frac{1}{\sqrt{2}}|² = \frac{1}{2}$

"""

# ╔═╡ 7b7d106f-871b-4354-8f58-f64818ab81f8
CX * CH * NQubit(
	Qubit(
		1/sqrt(2), 
		1/sqrt(2)
	),
	Qubit(
		1/sqrt(2),
		1/sqrt(2)
	)
) |> probabilities

# ╔═╡ 9e04d441-f2a5-41e7-8e78-2a9ef8044e0b
md"""
## World 2 - Other Gates
"""

# ╔═╡ 449c2878-7f8b-47a2-8acb-dbfebf818ac8
md"""
## World 3 - Oracles
"""

# ╔═╡ c666985a-b56b-454e-94fa-07383ccb9a5b
md"""
## World 4 - Basic Quantum Algorithms
"""

# ╔═╡ dd337f85-a24b-4408-b38b-7632a592cc61
md"""
## World 5 - Advanced Quantum Algorithms
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0-rc3"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╟─be266b88-0ee8-11ed-17f8-7b18be34e038
# ╠═c136d6c1-67c5-42d3-a039-895ebb6caf4e
# ╠═f21a6cda-9a7e-471c-b80e-f7afe3931e5d
# ╟─ebcbe195-638f-439d-8fa6-24df225134e2
# ╠═0810d2b4-8d69-4588-b063-24f5798bcf10
# ╠═cc136271-39bf-4a0a-a57d-56c9168c726f
# ╠═8a0e755f-205d-488f-b58e-b340c617ff98
# ╠═9f48d5d9-320c-4a82-9f88-960668887f55
# ╠═9fdab464-aff2-44ba-b058-ac6fa56293a9
# ╠═785460c9-e799-416f-8a34-8a0f7dabe1b5
# ╠═d1f14ea8-2ba6-40c6-a560-4efb08b647df
# ╠═40551fe4-3b70-46ea-847b-e0e0feef477c
# ╠═c6e5e683-5c8f-418f-ba04-f2dbdb06134f
# ╠═9cecbf6c-d6a5-4ec2-8647-56da7835c3f3
# ╟─c0a94f4f-d3de-486f-a882-28627cbfce1e
# ╟─f065fd22-20e4-48e8-b2b4-b88b7f0fae18
# ╠═0404d066-999c-4b1e-9434-010c487a0968
# ╠═5a52f5e3-b9aa-485a-9689-2d5d58b5cfe0
# ╠═7b5461bd-71bb-4155-b886-1285bc842078
# ╠═da1cba03-b50d-4f37-b94c-5f6538062041
# ╠═7f768046-8e1b-4429-9727-958ef902f86e
# ╠═bf204d31-c0ff-4313-abb7-4f65aacb1d1b
# ╠═3ce5ac59-22e0-4477-a5e0-4db2c97e12b3
# ╟─7589200e-711a-4c03-95d5-cdc422e82289
# ╠═3157d8e0-fb97-4774-b80c-3a455d9d558f
# ╠═66c08c17-f32e-44c6-8d60-63dc25172812
# ╠═74e48ac3-8743-4f30-a5c2-c127007902ad
# ╠═cccefe0c-95ca-40b3-8642-d3edd1a14081
# ╠═0761a159-7268-46ca-a9c7-b58efb2db7dc
# ╠═1a324973-1f51-4111-a776-8265f9ce092d
# ╟─8996891f-087a-44cc-87ed-d2f94ba09187
# ╠═356e92c0-66e9-49e1-9938-462c63b2f4bd
# ╠═3ce4fd45-3e37-4263-84ec-7ed04828edb4
# ╠═9db796f5-701e-42b5-81ea-78bae2c30132
# ╠═a9c85502-80d8-4363-84ad-8cd8e3486c15
# ╟─21b61975-72c5-4777-87e0-65ce4a365e7a
# ╟─5bc686c8-2d32-4e25-985e-ca95f4ba5da6
# ╟─3d8715d3-8421-4c73-9f93-a4703c3745fa
# ╠═3de435e2-0463-4859-a083-772e96e2a3f7
# ╟─57730711-ba0a-48a0-9c15-b627c6add2cb
# ╠═cfdc29e6-2aa6-4c54-8d43-05856a0aba7b
# ╟─5c9a6968-d36f-4219-90bf-cc0753b38457
# ╠═2a4310c1-4250-4c9c-a7e7-7f9d894c21d3
# ╟─b05eb619-9a20-42f0-bec8-6f3b0f37c34c
# ╠═4de05494-50e9-480c-96d5-5f5292c09b8d
# ╟─e36db4ac-d02a-49aa-8966-90428ddfa6b6
# ╠═eac3549b-b79b-4255-8747-ad202feaf9c9
# ╟─2d9180cd-79d0-4398-94b4-8e28c21ca24d
# ╠═525685e9-a15b-492a-bc18-4f81ad0414d9
# ╟─27a89157-745c-4e4d-958a-5886e812d284
# ╠═a59ef905-8c2b-480a-888f-87d8aad3ee03
# ╟─b2ceb23e-8eaa-4fd3-87fe-94d71978cfc0
# ╠═7b7d106f-871b-4354-8f58-f64818ab81f8
# ╟─9e04d441-f2a5-41e7-8e78-2a9ef8044e0b
# ╟─449c2878-7f8b-47a2-8acb-dbfebf818ac8
# ╟─c666985a-b56b-454e-94fa-07383ccb9a5b
# ╟─dd337f85-a24b-4408-b38b-7632a592cc61
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
