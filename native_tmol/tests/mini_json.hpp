// Minimal dependency-free JSON reader for the offline parity fixtures.
//
// A small recursive-descent parser (objects, arrays, numbers, strings, true/false/null)
// sufficient to read the tmol-webgpu fixture files (test/fixtures/*.json). It is test-only
// scaffolding: the energy kernels themselves never touch JSON. Kept deliberately small and
// reviewable rather than vendoring a full JSON library.

#pragma once

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace minijson {

struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    bool isNull() const { return type == Type::Null; }
    double num() const { return number; }
    int integer() const { return static_cast<int>(number); }
    bool boolv() const { return boolean; }
    const std::vector<Value>& array() const { return arr; }
    const Value& at(std::size_t i) const { return arr.at(i); }
    const Value& operator[](const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end()) throw std::out_of_range("json key not found: " + k);
        return it->second;
    }
    bool has(const std::string& k) const { return obj.find(k) != obj.end(); }
};

class Parser {
   public:
    explicit Parser(const std::string& text) : s_(text), n_(text.size()) {}

    Value parse() {
        skipWs();
        Value v = parseValue();
        skipWs();
        return v;
    }

   private:
    const std::string& s_;
    std::size_t n_;
    std::size_t i_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("json parse error at " + std::to_string(i_) +
                                 ": " + msg);
    }
    char peek() const { return i_ < n_ ? s_[i_] : '\0'; }
    char get() { return i_ < n_ ? s_[i_++] : '\0'; }
    void skipWs() {
        while (i_ < n_) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
                ++i_;
            else
                break;
        }
    }

    Value parseValue() {
        skipWs();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') {
            Value v;
            v.type = Value::Type::String;
            v.str = parseString();
            return v;
        }
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        return parseNumber();
    }

    Value parseObject() {
        Value v;
        v.type = Value::Type::Object;
        get();  // {
        skipWs();
        if (peek() == '}') {
            get();
            return v;
        }
        while (true) {
            skipWs();
            if (peek() != '"') fail("expected string key");
            std::string key = parseString();
            skipWs();
            if (get() != ':') fail("expected ':'");
            v.obj[key] = parseValue();
            skipWs();
            char c = get();
            if (c == ',') continue;
            if (c == '}') break;
            fail("expected ',' or '}'");
        }
        return v;
    }

    Value parseArray() {
        Value v;
        v.type = Value::Type::Array;
        get();  // [
        skipWs();
        if (peek() == ']') {
            get();
            return v;
        }
        while (true) {
            v.arr.push_back(parseValue());
            skipWs();
            char c = get();
            if (c == ',') continue;
            if (c == ']') break;
            fail("expected ',' or ']'");
        }
        return v;
    }

    std::string parseString() {
        if (get() != '"') fail("expected '\"'");
        std::string out;
        while (i_ < n_) {
            char c = s_[i_++];
            if (c == '"') return out;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        // Skip a \uXXXX escape (not used by the fixtures); emit '?'.
                        for (int k = 0; k < 4 && i_ < n_; ++k) ++i_;
                        out.push_back('?');
                        break;
                    }
                    default: fail("bad string escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    Value parseBool() {
        Value v;
        v.type = Value::Type::Bool;
        if (s_.compare(i_, 4, "true") == 0) {
            i_ += 4;
            v.boolean = true;
        } else if (s_.compare(i_, 5, "false") == 0) {
            i_ += 5;
            v.boolean = false;
        } else {
            fail("bad bool literal");
        }
        return v;
    }

    Value parseNull() {
        Value v;
        v.type = Value::Type::Null;
        if (s_.compare(i_, 4, "null") == 0)
            i_ += 4;
        else
            fail("bad null literal");
        return v;
    }

    Value parseNumber() {
        std::size_t start = i_;
        if (peek() == '-' || peek() == '+') get();
        while (i_ < n_) {
            char c = s_[i_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-')
                ++i_;
            else
                break;
        }
        if (i_ == start) fail("expected number");
        Value v;
        v.type = Value::Type::Number;
        v.number = std::strtod(s_.c_str() + start, nullptr);
        return v;
    }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace minijson
