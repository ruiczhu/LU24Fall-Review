# CDS528 Blockchain

## Module 1. Bitcoin Blockchain

### 1. Bitcoin Performance
1. Security: 50%+1 attack
2. Transaction Throughput: 7 transactions per second
3. Confirmation Latency: hours
4. Energy Consumption: Medium-size country / 0.5% of global energy consumption
5. Compute: Specialized hardware
6. Storage: Everyone stores everything
7. Communication: Everyone tx/rx everything

<p style="display: block;">
    <img src="image_182.png" alt="image_182"/>
</p>

### 2. Cryptographic data structures
<p style="display: block;">
    <img src="image_183.png" alt="image_183"/>
</p>

- *Structure of Block*

<p style="display: block;">
    <img src="image_184.png" alt="image_184"/>
</p>

- *Structure of Blockchain Header*

### 3. Permissioned and Permission-less Chains
<p style="display: block;">
    <img src="image_185.png" alt="image_185"/>
</p>

### 4. Consensus
1. Proof of Work(PoW)
2. Proof of Stake(PoS)
<p style="display: block;">
    <img src="image_186.png" alt="image_186"/>
</p>

### 5. Building Block of Blockchains
1. Platforms are applications built on networked computers
2. Basic building block:  **Decentralized Computer**
   1. Multiple untrusted computers interacting with one another, forming **consensus** on an ordered list of instructions
   2. A **virtual machine** interprets the instruction set
   3. A programming language and a corresponding compiler provide a forum for **decentralized applications (dApps)**

#### 5.1. Technical Components
1. Decentralized Computer
   1. Cryptographic data structures 
   2. Disk I/O and Database management 
   3. Memory management 
   4. Operating systems 
   5. Peer to peer networking 
   6. Consensus and distributed algorithms
2. Virtual Machine
   1. Reduced instruction set, incentives 
   2. General purpose programming language 
   3. Nearly all aspects of Computer Science

- <p style="display: block;">
    <img src="image_187.png" alt="image_187"/>
  </p>

### 6. Digital Signatures
1. Key generation: 
   - (Public Key, Private Key) = GenerateKeys(keysize)
2. Signature generation: 
   - Signature = Sign(PrivateKey, Message)
3. Signature verification:
   - Verify(PublicKey, Message, Signature)

### 7. Merkle Trees
1. 快速归纳和校验区块数据：
   - Merkle树可以将区块链中的数据分组进行哈希运算，向上不断递归运算产生新的哈希节点，最终只剩下一个Merkle根存入区块头中。这样，区块头只需包含根哈希值而不必封装所有底层数据，从而极大地提高了区块链的运行效率和可扩展性。
2. 支持简化支付验证协议（SPV）：
   - Merkle树使得在不运行完整区块链网络节点的情况下，也能够对交易数据进行检验。SPV钱包可以通过验证Merkle树中的部分节点来确认交易的真实性，而无需下载整个区块链。
3. 提高数据完整性和一致性：
   - Merkle树的任何底层数据的变动都会传递到其父节点，一直到树根。因此，通过比较Merkle树根的值，可以快速确定两个数据集是否一致。这种特性在分布式系统中尤为重要，因为它可以大大节省比对时间以及数据的传输量。
<p style="display: block;">
    <img src="image_188.png" alt="image_188"/>
</p>

### 8. Types of Network Architectures
1. Peer-to-Peer Network
   - <p style="display: block;">
       <img src="image_190.png" alt="image_190"/>
     </p>
2. Client-Server Network
   - <p style="display: block;">
       <img src="image_189.png" alt="image_189"/>
     </p>
    
    
### 9. Longest Chain Protocol
- attach the block to the leaf of the longest chain in the block tree

- Blockchain may have forks
  - because of network delays
  - because of adversarial action

### 10. Chain Growth
- Chain growth is 1 block per 10 minutes

### 11. Chain Quality (CQ)
- Chain quality is the number of blocks in the longest chain
- $$CQ = \frac{N}{M}$$
  - N: number of honest blocks in the longest chain
  - M: number of all blocks in the longest chain
- “最长链”指的是区块链中当前最长的、包含最多区块的链，而“诚实区块”则是指那些没有被篡改或非法创建的区块，它们遵循了区块链网络的规则和共识机制。
- 一个高链质量意味着最长链上绝大多数区块都是诚实且未被篡改的，这有助于增强网络的整体信任度和可靠性。相反，如果链质量较低，说明网络中存在较多的非法或篡改行为，这可能会损害区块链的完整性和可信度。
    
### 12. Double Spending Attack
#### 1. Adversary can point its block to an older part of the chain
**攻击者可以将其区块指向链的较旧部分**，意味着攻击者可能尝试将新生成的区块附加到区块链的较早部分，而不是最新的区块。这种行为可能导致以下问题：

- **Duplicate transaction inserted**：如果攻击者成功地将一个包含重复交易的区块插入到链的较旧部分，可能会导致双重支付问题，即同一笔交易被记录两次。

#### 2. Plausible Deniability
**合理否认**在区块链上下文中意味着，即使出现了问题，也很难确定其原因或责任归属。

##### 具体场景
- **Network latency（网络延迟）**：在分布式网络中，不同节点之间的信息传递会有延迟。这种延迟可能导致同一交易在不同节点上显示的时间不一致，从而增加了对区块顺序的不确定性。
- **An offline user will not know which block came earlier（离线用户不知道哪个区块更早）**：如果用户在离线状态下进行交易，他们无法实时获取最新的区块信息，当他们重新上线时，可能无法确定区块链的最新状态。
- **Blocks have no wall clock reference (time stamps)（区块没有实时时钟参考（时间戳））**：虽然区块通常包含生成时间戳，但这些时间戳并不具有权威性，不是由一个集中时钟提供的。这意味着在跨节点传输和验证时，时间戳可能会出现差异，进一步增加了对区块顺序的混淆。

<p style="display: block;">
    <img src="image_191.png" alt="image_191"/>
</p>

### 13. Selfish Mining Attack (自私挖矿攻击)

#### Overview (概述)
**Selfish Mining Attack** is a strategy targeting blockchain networks that use the Proof of Work (PoW) consensus mechanism. The core idea is that an attacker (or malicious mining pool) deliberately delays broadcasting newly mined blocks and constructs a private chain to cause a fork. (自私挖矿攻击是一种针对采用工作量证明共识机制（PoW）的区块链网络的攻击策略。这种攻击的核心思想是攻击者（或恶意矿池）故意延迟公布其新挖的区块，并构造一个私有链，从而引起链的分叉。)

#### Attack Process (攻击过程)

1. **Delaying Block Broadcast (延迟公布新区块)**
    - The attacker mines a new block but does not immediately broadcast it to the network. Instead, they retain these blocks and continue mining on their private chain. (攻击者在挖到新区块后，并不立即将其广播到网络中，而是选择保留这些区块，并在私有链上继续挖矿。)

2. **Constructing a Private Chain (构造私有分支)**
    - The attacker constructs a private chain based on the retained blocks and continues mining on this branch, hoping to achieve a longer chain. (攻击者基于保留的区块构造自己的私有链，并在这个分支上继续挖矿，以期望获得更长的链。)

3. **Choosing the Timing to Release Blocks (选择时机公布区块)**
    - When the length of the attacker’s private branch exceeds the longest public branch in the network, the attacker releases these blocks, causing their private branch to become the longest chain. (当攻击者的私有分支长度超过网络中的最长公开分支时，攻击者会选择公布这些区块，从而使自己的私有分支成为最长链。)

4. **Gaining Extra Rewards (获取额外奖励)**
    - Because the blockchain’s consensus mechanism typically regards the longest chain as valid, the attacker’s private branch will be accepted by the network, allowing the attacker to claim the mining rewards for these blocks. (由于区块链的共识机制通常认为最长链是有效的，因此攻击者公布的私有分支会被网络接受，从而使攻击者获得这些区块的挖矿奖励。)

<p style="display: block;">
    <img src="image_192.png" alt="image_192"/>
</p>

### 14. The DAO Attack
<p style="display: block;">
    <img src="image_193.png" alt="image_193"/>
</p>

#### 1. Reentrancy Attack

**The Reentrancy attack** is one of the most destructive attacks in the Solidity smart contract. (重入攻击是Solidity智能合约中最具破坏性的攻击之一。)

Examples:
- Uniswap/Lendf.Me hacks (April 2020) – $25 million, attacked by a hacker using a reentrancy. 
- The BurgerSwap hack (May 2021) – $7.2 million because of a fake token contract and a reentrancy exploit. 
- The SURGEBNB hack (August 2021) – $4 million seems to be a reentrancy-based price manipulation attack. 
- CREAM FINANCE hack (August 2021) – $18.8 million, reentrancy vulnerability allowed the exploiter for the second borrow. 
- Siren protocol hack (September 2021) – $3.5 million, AMM pools were exploited through reentrancy attack.

**A reentrancy attack occurs when a function makes an external call to another untrusted contract. Then the untrusted contract makes a recursive call back to the original function in an attempt to drain funds.** (重入攻击发生在一个函数对另一个不可信的合约进行外部调用时，然后这个不可信的合约递归调用回原始函数，试图耗尽资金。)

##### Example Scenario (示例场景)

<p style="display: block;">
    <img src="image_194.png" alt="image_194"/>
</p>

In this example, an attacker can exploit the reentrancy vulnerability as follows: (在这个例子中，攻击者可以按如下方式利用重入漏洞：)

<p style="display: block;">
    <img src="image_195.png" alt="image_195"/>
</p>

1. **Initial Withdrawal (初始提款)**: The attacker calls the `withdraw` function. (攻击者调用 `withdraw` 函数。)
2. **External Call (外部调用)**: The contract sends Ether to the attacker's contract. (合约将以太币发送到攻击者的合约。)
3. **Recursive Call (递归调用)**: The attacker's contract immediately calls the `withdraw` function again before the balance is updated. (攻击者的合约在余额更新之前立即再次调用 `withdraw` 函数。)
4. **Repeat (重复)**: This process can be repeated to drain funds until the balance is depleted. (这个过程可以重复，直到余额耗尽。)

##### Preventing Reentrancy Attacks (防止重入攻击)

To prevent reentrancy attacks, consider the following best practices: (为了防止重入攻击，请考虑以下最佳实践：)

1. **Update State Before External Calls (在外部调用之前更新状态)**:
   Always update the state variables before making an external call. (在进行外部调用之前始终更新状态变量。)
    - <p style="display: block;">
        <img src="image_196.png" alt="image_196"/>
      </p>

2. **Use `ReentrancyGuard` (使用 `ReentrancyGuard`)**:
   Implement the `ReentrancyGuard` modifier provided by OpenZeppelin to prevent reentrant calls. (使用OpenZeppelin提供的 `ReentrancyGuard` 修饰符来防止重入调用。)
    - <p style="display: block;">
        <img src="image_197.png" alt="image_197"/>
      </p>

3. **Avoid Using `call` for Value Transfers (避免使用 `call` 进行数值传输)**:
   Prefer using `transfer` or `send` for transferring Ether, as they have a fixed gas limit and reduce the risk of reentrancy. (优先使用 `transfer` 或 `send` 进行以太币传输，因为它们有固定的gas限制，减少了重入的风险。)


## Module 2. Scaling Blockchain

### 1. Bitcoin Rule
1. The mining difficulty is adjusted every 2016 blocks
   - $$
   \text{Next Difficulty} = \frac{\text{Previous Difficulty} \times {2016} \times {10} \text{minutes}} {\text{time to mine last 2016 blocks}}
   $$

2. Adopt the heaviest chain instead of the longest chain
   - The heaviest chain is the chain with the most proof of work
   - Chain difficulty = sum of block difficulties
   
3. Allow the difficulty to be adjusted only mildly every 2016 blocks
   - $$
   \frac{\text{Next Difficulty}}{\text{Previous Difficulty}} \in [0.25, 4]
   $$

4. Halving
    - Every 210,000 blocks, the reward is halved
    - $$
    \text{Reward} = 50 \times \frac{1}{2^{\text{Halving Count}}}
    $$
    - $$
   \text{Halving Count} = \frac{\text{Block Number}}{210,000}
    $$

5. Block size limit
   - 1MB

### 2. Throughput
$$
\text{Throughput} = \frac{{(1 - \beta)}\times{\lambda}}{1 + (1 - \beta)\times{\lambda}\times{\Delta}}\times{B} {\,}
$$
- $\lambda$: mining rate; can be controlled by setting mining target easy
- $\beta$: fraction of adversarial hash power; no control; Honest control is $1 - \beta$
- $\Delta$: network delay; proportional to block size 𝐵
- $B$: block size; can be controlled by allowing more transaction in a block

- so throughput
- $$
  \propto \frac{{(1 - \beta)}\times{\lambda}\times{\Delta}}{1 + (1 - \beta)\times{\lambda}\times{\Delta}}\times{B} {\,}
  $$
- Limited by $\lambda$ and $\Delta$

### 3. Trilemma
<p style="display: block;">
    <img src="image_198.png" alt="image_198"/>
</p>
1. Security
2. Decentralization
3. Scalability

### 4. Lightening network fee structure
- Base Fee = 1 satoshi
- Fee Rate = 0.5 satoshi per million - 0.00005%
- $$
  \text{Fee} = \text{Base Fee} + \text{Fee Rate} \times \text{Amount}
  $$
- Current bitcoin on-chain transaction fee = 6600 Satoshi
- Total number of participating nodes ~ 20K
- Total number of channels ~ 85K
- Total capacity ~ USD 100M ~ 5K BTC
- Highest capacity node ~ 650 BTC

### 5. Scaling solutions
1. Option 1: increase gas limit
   - Equivalent to increase block size in Bitcoin
   - It takes longer to transmit the block   reduced security
   - Would raise hardware requirements for full nodes
2. Option 2: sharding
   - Each node stores/executes part of the transactions
   - Requires L1 change
3. Option3: sidechains
   - Moves both computation and storage off-chain
   - Data availability issue

### 6. Desired properties
1. L2 solution
   - Lightweight protocol via smart contracts
   - No change in L1
2. Security and data availability
   - Reuse Ethereum security
   - Avoid introducing additional trust
3. EVM compatibility
   - Easy to migrate a Dapp from L1 to L2
   - Reduced Tx fees and increased Tx throughput compared to L1

### 7. Rollups
<p style="display: block;">
    <img src="image_199.png" alt="image_199"/>
</p>

- Execute transactions off-chain (outsource computation – just like sidechains)
- Report data on-chain in a compressed way (but enough to verify execution is correct)  -- so no data availability problem

Optimistic Rollups: pros and cons
<p style="display: block;">
    <img src="image_200.png" alt="image_200"/>
</p>

### 8. zk-Rollups

**ZK-SNARKs**（Zero-Knowledge Succinct Non-Interactive Argument of Knowledge）是一种加密证明，允许一方（证明者）向另一方（验证者）证明其知道某个秘密信息，而无需透露该信息的具体内容。以下是中英对照的解释。

#### Components (组成部分)
- **C: the state transition program**（C：状态转换程序）
    - This is the program that describes how the state changes from one state to another. （这是描述状态如何从一个状态变化到另一个状态的程序。）

- **x: pre-state, post-state**（x：前状态，后状态）
    - The `pre-state` is the state before any transactions occur, and the `post-state` is the state after all transactions have been applied. （`前状态` 是交易发生前的状态，而 `后状态` 是所有交易应用后的状态。）

- **w: all transactions**（w：所有交易）
    - This represents all the transactions that have taken place which cause the state transition. （这代表所有导致状态转换的交易。）

#### ZK Rollup
- **A ZK rollup coordinator generates a SNARK proof π that proves it knows the private transactions such that the post-state is correctly updated from the pre-state.**
    - **一个ZK汇总协调员生成一个SNARK证明π，证明其知道私有交易，使得后状态从前状态正确更新。**

    - **Detailed Explanation:**
        - The coordinator uses the state transition program  C  and the transactions  w  to compute the new state  x  from the pre-state to the post-state. （协调员使用状态转换程序  C  和交易  w  计算新状态  x ，从前状态到后状态。）
        - The SNARK proof π is a cryptographic proof that the coordinator knows the transactions  w  and has correctly applied them to transition the state from pre-state to post-state. This proof does not reveal the actual transactions  w , ensuring privacy. （SNARK证明π 是一个加密证明，表明协调员知道交易  w  并且已正确应用它们以将状态从前状态转换到后状态。此证明不揭示实际交易  w ，确保隐私。）

#### Validity proof example
<p style="display: block;">
    <img src="image_201.png" alt="image_201"/>
</p>

## 3. Module 3. Beyond Bitcoin

### 1. Data Availability Oracle
<p style="display: block;">
    <img src="image_202.png" alt="image_202"/>
</p>

### 2. Solution: Two Confirmation Rules
1. Availability-preserving rule
   - Remains live and safe under variable participation
   - Requires synchrony for liveness and safety
2. Finality-preserving rule
   - Remains safe under all conditions
   - Is live only under synchrony and fixed participation
<tip>Each rule generates its own ledger!</tip>

### 3. Finality Gadget
1. Two-layer design
   - Layer-one: Proof-of-Work Longest Chain
   - Layer-two: Committee-based BFT protocol
2. Longest chain protocol produces and confirms blocks
   - Works with variable participation
   - k-deep rule remains viable
3. BFT protocol independently confirms blocks
   - Confirms the same set of blocks as produced by PoW!
   - Switches on or off based on participation level

#### 1. Checkpointing
<p style="display: block;">
    <img src="image_203.png" alt="image_203"/>
</p>
- Rules of Checkpointing
<p style="display: block;">
    <img src="image_204.png" alt="image_204"/>
</p>

#### 2. Forensic Support
<p style="display: block;">
    <img src="image_205.png" alt="image_205"/>
</p>

#### 3. Privacy and Programmability
<p style="display: block;">
    <img src="image_206.png" alt="image_206"/>
</p>

### 4. Bridging
<p style="display: block;">
    <img src="image_207.png" alt="image_207"/>
</p>

- Construction: Privacy Bridge
<p style="display: block;">
    <img src="image_208.png" alt="image_208"/>
</p>

### 5. DeFi
- DeFi is tokenized finance on decentralized platforms

### 6. Tokenization
- Converting a tangible/intangible asset into a digital format
- Can be fungible (“currency”) or not (“an image or a video clip”)
- Awfully similar to securitization
  - Key is the missing trusted middle party

### 7. Tokenized Finance
- Commerce  – buying, selling
- Market places – exchanges
- Options, derivatives – financial instruments
- Borrowing, lending – banks 

### 9. Nine elements of DeFi
1. Token transfers: native blockchain transactions 
2. Market making via smart contracts 
3. Oracles: importing external data 
4. Borrow/Lending: banking functionality 
5. Cross border finance: bridges, wrapped tokens 
6. Stable coins: tying tokens to fiat 
7. Synthetics and Perpetuals: self-adapting financial instruments 
8. NFT: digital collectibles 
9. DAO: tokenized governance