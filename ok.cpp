//DBL_EPSILON
//__builtin_ctz(i) 0000001
//__builtin_clz(i)10000000
//__builtin_popcount(n)
//freopen("input.txt", "r", stdin); freopen("output.txt", "w", stdout);
//leab year month 2 feb +1 366 not 365
int day[] = { 0,31,28,31,30,31,30,31,31,30,31,30,31 };
priority_queue<int,vector<int>,greater<int>> p2; // top() = smallest
to right ->rotate(vect.begin(), vect.begin()+vect.size() + k, vect.end());
rotate(vect.begin(), vect.begin() + k, vect.end());
cout<<(ll)ceil((-1+sqrt(1+8*x))/2)<<endl;
scanf("%d.%d.%d", &a, &b, &c);

#define fix(n,k) (n%k+k)%k
ll ceil(ll a, ll b) { return (a + b - 1) / b; }

auto distance = [&](pair<double,double> a, pair<double,double> b)->double {
    return sqrt((a.first - b.first) * (a.first - b.first) + (a.second - b.second) * (a.second - b.second));
};


ll add(ll x, ll y) {
    return ((x + y) % MOD + MOD) % MOD;
}

vector<int> ok(vector<int>a,vector<int>b) {
    vector<int>ret(4);
    int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3],
        x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];
    x1 = max(x1, x3); x2 = min(x2, x4);y1 = max(y1, y3);y2 = min(y2, y4);
    ret[0] = x1, ret[1] = y1, ret[2] = x2, ret[3] = y2;
    if (x1 >= x2 || y1 >= y2)ret[0] = -1;//failed
    return ret;
}

<<fixed<<setprecision(6) <<

//mirror grid 0 90 180 270
int cnt=((s[i][j]=='1')+(s[j][n-i-1]=='1')+(s[n-i-1][n-j-1]=='1')+(s[n-j-1][i]=='1'));

//(primary diagonal and secondary diagonal). map it by (i-j), (i+j);
// i==j , i+j==n;

//Simple way of eliminating all duplicates in a vector is to make use of std::unique.
sort(all(vect));
vect.erase(unique(all(vect)), vect.end());


if(b<c||a>d) cout<<-1;
else cout<<max(a,c)<<' '<<min(b,d);

bool palindrome(string s){
    for(int i=0;i<s.size()/2;i++)
        if(s[i]!=s[s.size()-1-i])return 0;
    return 1;
}


bool is_prime(long long n){
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (long long i = 5; i * i <= n; i += 6)
        if (n % i == 0 || n % (i + 2) == 0) return false;
    return true;
}

int dx[] = {+0, +0, -1, +1, +1, +1, -1, -1};
int dy[] = {-1, +1, +0, +0, +1, -1, +1, -1};
bool valid(int i,int j){return (i>=0&&i<n&&j>=0&&j<m);}
auto  valid = [&](int i, int j) { return (i >= 0 && i < n && j >= 0 && j < m); };

ll gcd(ll a, ll b) { return ((b == 0) ? a : gcd(b, a % b)); }
ll lcm(ll a, ll b) { return (b / gcd(a, b)) * a; }


ll getsum(int a, int b) {
    if (a < b)swap(a, b);
    if(b>a)return 0;
    return ((a + b)*(a - b + 1)) / 2;
}


struct str{
    int val,x,y;
};
class Compare {
public:
    bool operator()(str below, str above){
        return (below.val>above.val);
    }
};
priority_queue<str,vector<str>,Compare>qu;




map<int,int> primeFactors(int n){
    map<int,int>mp;
    while (n % 2 == 0)mp[2]++,n = n/2;
    for (int i = 3; i <= sqrt(n); i = i + 2)
        while (n % i == 0)mp[i]++,n = n/i;
    if (n > 2)mp[n]++;
    return mp;
}

set<int>s;
int n;cin>>n;
for(int i=1;i<=sqrt(n);i++){
    if(n%i==0){
        s.insert(i);
        s.insert(n/i);
    }
}
for(auto it:s)cout<<it<<' ';



//a=factorila[a], b=MOD-2;
ll binpow(ll a, ll b) {
    ll res = 1;
    a %= MOD;
    while (b)
    {
        if (b & 1)
            res = (res * a) % MOD;
        a = (a * a) % MOD;
        b >>= 1;
    }
    return res;
}

fact.resize(n + 1);
invfact.resize(n + 1);
fact[0] = fact[1] = 1;
for (int i = 2; i <= n; i++) {
    fact[i] = fact[i - 1] * i;
    fact[i] %= MOD;
}
invfact[0] = 1;
for (int i = 1; i <= n; i++) {
    invfact[i] = binpow(fact[i], MOD - 2);
}

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#define ordered_set tree<int, null_type, less_equal<int>, rb_tree_tag, tree_order_statistics_node_update>

using namespace __gnu_pbds;

//#define ordered_set tree<pair<int, int>, null_type,less<pair<int, int>>, rb_tree_tag,tree_order_statistics_node_update>
/*
    A.order_of_key(6)//serch lower_bound of key
    *A.find_by_order(3)
    *A.lower_bound(6)==upper bound
    *A.upper_bound(6)==lower bound
    A.erase(idx)
    ok.erase(ok.upper_bound(7));
*/

#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace __gnu_pbds;
template<class T>using ordered_set =tree<T,null_type,less<T>,rb_tree_tag,tree_order_statistics_node_update>;

/*
order_of_key(k) number of items strictly smaller than k
find_by_order(k) kth element in aset 0-base
*/

template<class T>using ordered_set = tree<T, null_type, less_equal <T>, rb_tree_tag, tree_order_statistics_node_update>;
s.erase(s.upper_bound(1));

// generating random numbers from -4 to 4 size n
vector<int>v;
void b(int i){
    if(i==n){
        if(valid(v)){
            for(auto&it:v){
                cout<<it<<' ';
            }
            cout<<'\n';
        }
        return;
    }
    for(int j=-4;j<=4;j++){
        v.push_back(j);
        b(i+1);
        v.pop_back();
    }
}


//generat all subsequance
vector<int>v={1,2,3,4,5,6}, a;
vector<vector<int>>test;
void solve(int i){
    if(i==v.size()) {
        // do ur test here;
        test.push_back(a);
        return;
    }
    a.push_back(v[i]);
    solve(i+1);
    a.pop_back();
    solve(i+1);

}

//generate all permutations
vector<int>v={1,2,3,4,5};
do{
    // do ur test here;


}while(next_permutation(all(v)));




vector<vector<ll>>v(n+1,vector<ll>(m+1,0));
for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)
        cin>>v[i][j];

for(int i=1;i<=n;i++)
    for(int j=1;j<=m;j++)v[i][j]+=v[i-1][j]+v[i][j-1]-v[i-1][j-1];
//the lower left corner and the upper right
int x1, x2, y1, y2; cin >> x1 >> y1 >> x2 >> y2;
cout<<v[x2][y2]-v[x2][y1-1]-v[x1-1][y2]+v[x1-1][y1-1]<<endl;



auto solve = [&](auto solve, int i, bool leading, bool greater, int sum, int sm)->int {
    if (i == ok.size())return (sum == 0 && sm == 0);
    if (~dp[i][leading][greater][sum][sm])return dp[i][leading][greater][sum][sm];
    int ret = 0;
    for (int j = 0; j <= 9; j++) {
        if (!greater && ok[i] < j + '0')break;
        ret += solve(solve, i + 1, leading | (j > 0), greater | (ok[i] != (j + '0')), (sum * 10 + j) % m, (sm + j) % m);
    }
    return dp[i][leading][greater][sum][sm] = ret;
};


void getAllSubMasks(int mask) {

	for(int subMask = mask ; subMask ; subMask = (subMask - 1) & mask)
		printNumber(subMask, 32);	// this code doesn't print 0

	// For reverse: ~subMask&mask = subMask^mask
}



// using hashing
int count_unique_substrings(string const& s) {
    int n = s.size();

    const int p = 31;
    const int m = 1e9 + 9;
    vector<long long> p_pow(n);
    p_pow[0] = 1;
    for (int i = 1; i < n; i++)
        p_pow[i] = (p_pow[i-1] * p) % m;

    vector<long long> h(n + 1, 0);
    for (int i = 0; i < n; i++)
        h[i+1] = (h[i] + (s[i] - 'a' + 1) * p_pow[i]) % m;

    int cnt = 0;
    for (int l = 1; l <= n; l++) {
        set<long long> hs;
        for (int i = 0; i <= n - l; i++) {
            long long cur_h = (h[i + l] + m - h[i]) % m;
            cur_h = (cur_h * p_pow[n-i-1]) % m;
            hs.insert(cur_h);
        }
        cnt += hs.size();
    }
    return cnt;
}





__int128 read() {
    __int128 x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}
void print(__int128 x) {
    if (x < 0) {
        putchar('-');
        x = -x;
    }
    if (x > 9) print(x / 10);
    putchar(x % 10 + '0');
}
__int128 n=read();

//seive prime factors
void init() {
    fast;
    for (int i = 2; i < N; ++i) if (divs[i].empty()) for (int j = i; j < N; j += i) divs[j].push_back(i);
}


// finding b if it is substr from a;
void KMP(string a, string b){
    vector<int>prfx(b.size());
    for(int i=1, k=0;i<b.size();i++){
        while(k>0&&b[k]!=b[i])
            k=prfx[k-1];

        if(b[i]==b[k]) prfx[i]=++k;
        else prfx[i]=k;
    }
    for(int i=0,k=0;i<a.size();i++){
        while(k>0&&a[i]!=b[k])
            k=prfx[k-1];
        if(a[i]==b[k]) k++;

        if(k==b.size()){
            //found
            k=prfx[k-1];
        }
    }
}




int idx;
vector<vector<int>>v, scc_vec;
vector<int>vis,lowlink,instack,comp;
stack<int>st;
stack<pair<int,int>>component;
set<int>artpoints;
bool root;

void tarjan(int i) {
	lowlink[i] = vis[i] = instack[i] = ++idx;
	st.push(i);
	for (auto it : v[i]) {
		if (!vis[it]) {
			tarjan(it);
			//if (lowlink[it] > lowlink[i])
			lowlink[i] = min(lowlink[it], lowlink[i]);
		}
		else if (instack[it]) {
			lowlink[i] = min(lowlink[i], vis[it]);
		}
	}

	if (lowlink[i] == vis[i]) {
		scc_vec.push_back(vector<int>());
		int x = -1;
		while (x != i) {
			x = st.top(), st.pop(), instack[x] = 0;
			scc_vec.back().push_back(x);
			comp[x]=scc_vec.size()-1;
		}
	}
}

void init(int n){
    v.clear();scc_vec.clear();
    vis.clear();lowlink.clear();
    instack.clear();comp.clear();
    artpoints.clear();root=0;

    st=stack<int>();
    component=stack<pair<int,int>>();
    v.resize(n+1);vis.resize(n+1);
    lowlink.resize(n+1);comp.resize(n+1);
    instack.resize(n+1);
}
void scc() {
	int n,m;
	cin >> n >> m;
	init(n);
	for (int i = 0; i < m; i++) {
		int x, y; cin >> x >> y;
		v[x].push_back(y);
	}

	for (int i = 0; i < n; i++) {
		if (!vis[i]) tarjan(i);
	}
}

void computeCompGraph() {
	int csz = scc_vec.size(), cntSrc = csz, cntSnk = csz;
	vector<int>outDeg(csz), inDeg(csz);
	vector<vector<int>>dagList(csz);//will contain duplicates

	for (int i = 0; i < v.size(); i++)
		for (int j = 0; j < v[i].size(); j++) {
			int k = v[i][j];
			if (comp[k] != comp[i]) {
				dagList[comp[i]].push_back(comp[k]);	//reverse
				if (!(inDeg[comp[k]]++))		cntSrc--;
				if (!(outDeg[comp[i]]++))		cntSnk--;
			}
			else
				;// this edge is for a component comp[k]
		}

	 //Min edges to convert DAG to one cycle
	if (scc_vec.size() == 1)
		cout << "0\n";
	else {
		cout << max(cntSrc, cntSnk) << "\n";
	}
}

void bridges(int i,int p) {
	lowlink[i] = vis[i] = ++idx;
	for (auto it : v[i]) {
		if (!vis[it]) {
			bridges(it, i);
			lowlink[i] = min(lowlink[i], lowlink[it]);
			if (lowlink[it] == vis[it]) {
				cout << it << ' ' << i << endl;
			}
		}
		else if (it != p) lowlink[i] = min(lowlink[i], vis[it]);
	}
}

void Articulation(int i,int p=-1) {//p=-1, when u call it;
    if(p==-1)root=0;
	lowlink[i] = vis[i] = ++idx;
	for (auto it : v[i]) {
        if(it!=p&&vis[i]>vis[it])component.push({i,it});
		if (!vis[it]) {
			Articulation(it, i);
			lowlink[i] = min(lowlink[i], lowlink[it]);
			if(lowlink[it]>=vis[i]){
                if(p==-1&&!root)root=1;
                else artpoints.insert(i);
                int ok=0;
                pair<int,int>temp;
                do{
                    ok++;
                    temp=component.top();component.pop();
                    cout<<temp.first<<' '<<temp.second<<endl;
                }while(temp.first!=i||temp.second!=it);
                if(ok==1) cout<<temp.second<<' '<<temp.first<<endl; //cycle from one
                cout<<endl<<endl;
			}
		}
		else if (it != p) lowlink[i] = min(lowlink[i], vis[it]);
	}
}

struct str{
    int idx,w;
    str(int idx, int w):idx(idx),w(w){}
    bool operator <(const str &s) const{
        return w>e.w;
    }
};


vector<vector<pair<int,int>>>v;//second =weight
void dijkstra(int s,vector<vector<pair<int,int>>>v){
    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>>pq;
    vector<int>dist(v.size(),1e8);
    pq.push({0,s});
    dist[s]=0;
    while(!pq.empty()){
        auto temp=pq.top();pq.pop();
        //if(temp.second==ok) return temp.first;
        if(temp.first>dist[temp.second])continue;
        for(auto it:v[temp.second]){
            if(dist[it.first]>dist[temp.second]+it.second){
                dist[it.first]=dist[temp.second]+it.second;
                pq.push({dist[it.first],it.first});
            }
        }
    }
}


for (int k = 0; k < n; k++)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            v[i][j] = min(v[i][j], v[i][k] + v[k][j]);

if (AdjMat[i][k] + AdjMat[k][j] < AdjMat[i][j]) {
    AdjMat[i][j] = AdjMat[i][k] + AdjMat[k][j];
    p[i][j] = p[k][j]; // update the parent matrix
}
void printPath(int i, int j) {
if (i != j) printPath(i, p[i][j]);
printf(" %d", j);
}


// to use MST make vector<str>{i,j,cost}; and sort it;
struct str {
    ll x, y, cost;
    bool operator <(str const& other) {
        return cost < other.cost;
    }
};
vector<int>parent, sz;int forestes;
void init(int n) {
    forestes=n;
    parent.clear(); sz.clear();
    parent.resize(n + 1);
    sz.resize(n + 1, 1);
    for (int i = 0; i <= n; i++)parent[i] = i;
}

int get(int i) {
    if (parent[i] == i)return i;
    return parent[i] = get(parent[i]);
}

bool link(int a, int b) {
    int x = get(a), y = get(b);
    if (x == y)return 0;
    if (sz[x] < sz[y])swap(x, y);
    parent[y] = x, sz[x] += sz[y];forestes--;
    return 1;
}




int calcMex(set<int> s) {
    int cur = 0;
    while (s.find(cur) != s.end()) cur++;
    return cur;
}
int dp[500005];
int calcGrundy(int n) {

    if (~dp[n]) return dp[n];

    set<int> sub_nimbers;

    //restrictions
    for (int i = l; i <= min(n, r); i++)
        if (n >= i)
            sub_nimbers.insert(calcGrundy(n - i));

    return dp[n] = calcMex(sub_nimbers);
}





//mo's algorithm 0 based
void remove(int idx) {

}
void add(int idx) {

}
ll get_answer() {
    return ans;
}
int block_size;
struct Query {
    int l, r, idx;
    bool operator<(Query other) const
    {
        return make_pair(l / block_size, r) <
            make_pair(other.l / block_size, other.r);
    }
};
vector<ll> mo_s_algorithm(vector<Query> queries) {
    vector<ll> answers(queries.size());
    sort(queries.begin(), queries.end());
    // 0 based x-1, y-1
    int cur_l = 0;
    int cur_r = -1;
    for (Query q : queries) {
        while (cur_l > q.l)cur_l--,add(cur_l);
        while (cur_r < q.r)cur_r++,add(cur_r);
        while (cur_l < q.l)remove(cur_l), cur_l++;
        while (cur_r > q.r) remove(cur_r),cur_r--;
        answers[q.idx] = get_answer();
    }
    return answers;
}

void update(int root=1, int tree_interval_left=0, int  tree_interval_right=n-1, int l, int r,int val) {
    if (tree_interval_left >= l && tree_interval_right <= r)
        tree[root]=val;
    if (l > tree_interval_right || tree_interval_left > r) return 0;
    int mid = (tree_interval_left + tree_interval_right) / 2;
    solve(2 * root, tree_interval_left, mid, l, r);
    solve(2 * root + 1, mid + 1, tree_interval_right, l, r);
    tree[root]=tree[2 * root]+tree[2 * root+1];
}

ll solve(int root=1, int tree_interval_left=0, int  tree_interval_right=n-1, int l, int r) {
    if (tree_interval_left >= l && tree_interval_right <= r)
        return tree[root];
    if (l > tree_interval_right || tree_interval_left > r) return 0;
    int mid = (tree_interval_left + tree_interval_right) / 2;
    return  solve(2 * root, tree_interval_left, mid, l, r) + solve(2 * root + 1, mid + 1, tree_interval_right, l, r);
}

while (__builtin_popcount(n) != 1)n++,v.push_back(0);
tree.resize(2 * n + 1);
for (int i = 0; i < n; i++)tree[n + i] = v[i];
for (int i = n-1; i >=0; i--) tree[i] = tree[2 * i] + tree[2 * i + 1];



//https://vjudge.net/contest/463569#problem/O
//using vector use vector.push_back(segment_tree<int>(ok));

template<typename T>
class segment_tree {
#define LEFT (idx<<1)
#define RIGHT (idx<<1|1)
#define MID ((start+end)>>1)
    int n;
    vector<T>tree, lazy;

    T merge(const T& left, const T& right) {
        return gcd(left, right);
        //return { left.x + right.x, max(left.prfx, left.x + right.prfx), max(right.suf, right.x + left.suf),max({right.mx,left.mx,right.prfx + left.suf}) };
    }
    inline void pushdown(int idx, int start, int end) {
        if (!lazy[idx])return;
        tree[idx] += lazy[idx];
        if (start != end) {
            lazy[LEFT] += lazy[idx];
            lazy[RIGHT] += lazy[idx];
        }
        lazy[idx] = 0;
    }
    inline void pushup(int idx) {
        tree[idx] = merge(tree[LEFT], tree[RIGHT]);
    }
    void build(int idx, int start, int end, const vector<T>& arr) {
        if (start == end) {
            tree[idx] = arr[start];
            return;
        }
        build(LEFT, start, MID, arr);
        build(RIGHT, MID + 1, end, arr);
        pushup(idx);
    }
    T query(int idx, int start, int end, int from, int to) {
        pushdown(idx, start, end);
        if (from <= start && end <= to)return tree[idx];
        if (to <= MID)return query(LEFT, start, MID, from, to);
        if (MID < from)return query(RIGHT, MID + 1, end, from, to);
        return merge(query(LEFT, start, MID, from, to), query(RIGHT, MID + 1, end, from, to));
    }
    void update(int idx, int start, int end, int lq, int rq, const T& val) {
        pushdown(idx, start, end);
        if (rq < start || end < lq)return;
        if (lq <= start && end <= rq) {
            lazy[idx] += val;
            pushdown(idx, start, end);
            return;
        }
        update(LEFT, start, MID, lq, rq, val);
        update(RIGHT, MID + 1, end, lq, rq, val);
        pushup(idx);
    }
public:
    segment_tree(int n) :n(n), tree(n << 2), lazy(n << 2) {
    }
    segment_tree(const vector<T>& v) {
        n = v.size();
        tree = vector<T>(n << 2);
        lazy = vector<T>(n << 2);
        build(1, 0, n-1 , v);
    }
    T query(int l, int r) {
        return query(1, 0, n-1, l, r);
    }
    void update(int l, int r, const T& val) {
        update(1, 0, n-1, l, r, val);
    }
#undef LEFT
#undef RIGHT
#undef MID

};

bool negativeCycle(vector<vector<pair<int,int>>>v) {
    int n = v.size()-1;
    vector<int>dist(n+1, 1e9); dist[1] = 0;
    for (int i = 0; i < n-1; i++) {
        for (int j = 1; j <= n; j++) {
            for (auto it : v[j]) {
                dist[it.first] = min(dist[it.first],it.second + dist[j]);
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        for (auto it : v[i]) {
            if (dist[it.first] > it.second + dist[i])return 1;
        }
    }
    return 0;
}


//LCA ()<- two nodes ;;
class Least_common_ancestor {
    int n;
    vector<vector<int>>dp;
    vector<int>depth; int LOG = 30;
    void init(vector<vector<int>>v) {
        depth.clear(); dp.clear();
        int n = v.size(); depth.resize(n);
        dp = vector<vector<int>>(n, vector<int>(32));
        auto dfs = [&](auto dfs, int i, int p = 0)->void {
            for (auto it : v[i]) {
                if (it == p)continue;
                dp[it][0] = i; depth[it] = depth[i] + 1;
                for (int j = 1; j < LOG; j++) {
                    dp[it][j] = dp[dp[it][j - 1]][j - 1];
                }
                dfs(dfs, it, i);
            }
        };
        dfs(dfs, 0);//1 based or not
    }
    int LCCA(int a, int b) {
        if (depth[a] > depth[b])swap(a, b);
        int k = depth[b] - depth[a];
        for (int i = 0; i < LOG; i++) {
            if (k & (1 << i))
                b = dp[b][i];
        }
        if (a == b)return a;
        for (int i = LOG - 1; i >= 0; i--) {
            if (dp[a][i] != dp[b][i]) {
                a = dp[a][i];
                b = dp[b][i];
            }
        }
        return dp[a][0];
    }
    int SB(int a, int b) {
        return depth[a] + depth[b] - 2 * depth[LCA(a, b)];
    }
public:
    Least_common_ancestor(vector<vector<int>>v) {
        n = v.size();
        init(v);
    }
    int LCA(int a, int b) {
        return LCCA(a, b);
    }

};

//EulerPhi(N):: Count the number of positive integers < N that are relatively prime
//to N. Recall: Two integers a and b are said to be relatively prime (or coprime) if
//gcd(a, b) = 1
ll EulerPhi(ll N) {//
    ll PF_idx = 0, PF = primes[PF_idx], ans = N; // start from ans = N
    while (PF * PF <= N) {
    if (N % PF == 0) ans -= ans / PF; // only count unique factor
    while (N % PF == 0) N /= PF;
    PF = primes[++PF_idx];
    }
    if (N != 1) ans -= ans / N; // last factor
    return ans;
}

/*
a*xx+b*yy=m;
since we got x0,y0,d;//first solution;
a*x+b*y=d;
a*x+b*y=d;
we can get xx&&yy from the first equation
then any xx=xx+(y/d)*n;
then any yy=yy+(x/d)*n;
*/
int x,y,d;
void extendedEuclid(int a, int b) {
    if (b == 0) { x = 1; y = 0; d = a; return; } // base case
    extendedEuclid(b, a % b); // similar as the original gcd
    int x1 = y;
    int y1 = x - (a / b) * y;
    x = x1;
    y = y1;
}

If the perimeter of a polygon is given, then its area can be calculated using the formula: Area = (Perimeter × apothem)/2. In this formula,
 the apothem should also be known or it can be calculated with the help of the formula, Apothem = [(length of one side)/{2 ×(tan(180/number of sides))}].

C(n, 0) = C(n, n) = 1 // base cases.
C(n, k) = C(n − 1, k − 1) + C(n − 1, k) // take or ignore an item, n>k> 0.
https://oeis.org/


double determinantOfMatrix(double mat[3][3])
{
    double ans;
    ans = mat[0][0] * (mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
          - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
          + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0]);
    return ans;
}

// This function finds the solution of system of
// linear equations using cramer's rule
void findSolution(double coeff[3][4])
{
    // Matrix d using coeff as given in cramer's rule
    double d[3][3] = {
        { coeff[0][0], coeff[0][1], coeff[0][2] },
        { coeff[1][0], coeff[1][1], coeff[1][2] },
        { coeff[2][0], coeff[2][1], coeff[2][2] },
    };
    // Matrix d1 using coeff as given in cramer's rule
    double d1[3][3] = {
        { coeff[0][3], coeff[0][1], coeff[0][2] },
        { coeff[1][3], coeff[1][1], coeff[1][2] },
        { coeff[2][3], coeff[2][1], coeff[2][2] },
    };
    // Matrix d2 using coeff as given in cramer's rule
    double d2[3][3] = {
        { coeff[0][0], coeff[0][3], coeff[0][2] },
        { coeff[1][0], coeff[1][3], coeff[1][2] },
        { coeff[2][0], coeff[2][3], coeff[2][2] },
    };
    // Matrix d3 using coeff as given in cramer's rule
    double d3[3][3] = {
        { coeff[0][0], coeff[0][1], coeff[0][3] },
        { coeff[1][0], coeff[1][1], coeff[1][3] },
        { coeff[2][0], coeff[2][1], coeff[2][3] },
    };

    // Calculating Determinant of Matrices d, d1, d2, d3
    double D = determinantOfMatrix(d);
    double D1 = determinantOfMatrix(d1);
    double D2 = determinantOfMatrix(d2);
    double D3 = determinantOfMatrix(d3);
    printf("D is : %lf \n", D);
    printf("D1 is : %lf \n", D1);
    printf("D2 is : %lf \n", D2);
    printf("D3 is : %lf \n", D3);

    // Case 1
    if (D != 0) {
        // Coeff have a unique solution. Apply Cramer's Rule
        double x = D1 / D;
        double y = D2 / D;
        double z = D3 / D; // calculating z using cramer's rule
        printf("Value of x is : %lf\n", x);
        printf("Value of y is : %lf\n", y);
        printf("Value of z is : %lf\n", z);
    }
    // Case 2
    else {
        if (D1 == 0 && D2 == 0 && D3 == 0)
            printf("Infinite solutions\n");
        else if (D1 != 0 || D2 != 0 || D3 != 0)
            printf("No solutions\n");
    }
}

double coeff[3][4] = {
        { 2, -1, 3, 9 },
        { 1, 1, 1, 6 },
        { 1, -1, 1, 2 },
    };

    findSolution(coeff);
https://vjudge.net/contest/571731#problem/F

vector<pair<int, int>>v(3);
for (int i = 0; i < 3; i++)cin >> v[i].first >> v[i].second;

double coeff[3][4] = {
{ v[0].first*v[0].first, v[0].first, 1, v[0].second},
{ v[1].first * v[1].first, v[1].first, 1, v[1].second },
{ v[2].first * v[2].first, v[2].first, 1, v[2].second },
};
findSolution(coeff);//x,y,z->a,b,c   ->ax^2+bx+c;


void findCircle(int x1, int y1, int x2, int y2, int x3, int y3)
{
    int x12 = x1 - x2;
    int x13 = x1 - x3;

    int y12 = y1 - y2;
    int y13 = y1 - y3;

    int y31 = y3 - y1;
    int y21 = y2 - y1;

    int x31 = x3 - x1;
    int x21 = x2 - x1;

    // x1^2 - x3^2
    int sx13 = pow(x1, 2) - pow(x3, 2);

    // y1^2 - y3^2
    int sy13 = pow(y1, 2) - pow(y3, 2);

    int sx21 = pow(x2, 2) - pow(x1, 2);
    int sy21 = pow(y2, 2) - pow(y1, 2);

    int f = ((sx13) * (x12)
        +(sy13) * (x12)
        +(sx21) * (x13)
        +(sy21) * (x13))
        / (2 * ((y31) * (x12)-(y21) * (x13)));
    int g = ((sx13) * (y12)
        +(sy13) * (y12)
        +(sx21) * (y13)
        +(sy21) * (y13))
        / (2 * ((x31) * (y12)-(x21) * (y13)));

    int c = -pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1;

    // eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    // where centre is (h = -g, k = -f) and radius r
    // as r^2 = h^2 + k^2 - c
    int h = -g;
    int k = -f;
    int sqr_of_r = h * h + k * k - c;

    int r = sqrt(sqr_of_r);
    // h,k,r
    x = h, y = k;
    rr = r;
}
